/**
 * @module ol/renderer/webgl/VectorLayer
 */
import ViewHint from '../../ViewHint.js';
import {assert} from '../../asserts.js';
import {listen, unlistenByKey} from '../../events.js';
import {buffer, createEmpty, equals} from '../../extent.js';
import BaseVector from '../../layer/BaseVector.js';
import {
  getTransformFromProjections,
  getUserProjection,
  toUserExtent,
  toUserResolution,
} from '../../proj.js';
import MixedGeometryBatch from '../../render/webgl/MixedGeometryBatch.js';
import VectorStyleRenderer from '../../render/webgl/VectorStyleRenderer.js';
import {colorDecodeId} from '../../render/webgl/encodeUtil.js';
import {createPostProcessDefinition} from '../../render/webgl/textUtil.js';
import VectorEventType from '../../source/VectorEventType.js';
import {
  apply as applyTransform,
  create as createTransform,
  makeInverse as makeInverseTransform,
  multiply as multiplyTransform,
  setFromArray as setFromTransform,
  translate as translateTransform,
} from '../../transform.js';
import {getUid} from '../../util.js';
import {
  create as createMat4,
  fromTransform as mat4FromTransform,
} from '../../vec/mat4.js';
import {DefaultUniform} from '../../webgl/Helper.js';
import WebGLRenderTarget from '../../webgl/RenderTarget.js';
import WebGLLayerRenderer from './Layer.js';
import {getWorldParameters} from './worldUtil.js';

export const Uniforms = {
  ...DefaultUniform,
  RENDER_EXTENT: 'u_renderExtent', // intersection of layer, source, and view extent
  PATTERN_ORIGIN: 'u_patternOrigin',
  GLOBAL_ALPHA: 'u_globalAlpha',
  TEXT_OVERLAY_TEXTURE: 'u_textOverlay',
  TEXT_OVERLAY_MATRIX: 'u_textOverlayMatrix',
};

const DEFAULT_TEXT_RENDER_THROTTLE_MS = 120;
const MIN_TEXT_RENDER_THROTTLE_MS = 16;
const MAX_TEXT_RENDER_THROTTLE_MS = 2000;
const TARGET_DUTY_CYCLE = 0.25;
const CRITICAL_FPS_THRESHOLD = 45;
const TEXT_RENDER_THROTTLE_SMOOTHING = 0.2;

function nowMs() {
  return typeof performance !== 'undefined' && performance.now
    ? performance.now()
    : Date.now();
}

/**
 * @typedef {import('../../render/webgl/VectorStyleRenderer.js').StyleShaders} StyleShaders
 */
/**
 * @typedef {import('../../style/flat.js').FlatStyleLike | Array<StyleShaders> | StyleShaders} LayerStyle
 */

/**
 * @typedef {Object} Options
 * @property {string} [className='ol-layer'] A CSS class name to set to the canvas element.
 * @property {LayerStyle} style Flat vector style; also accepts shaders
 * @property {Object<string, number|Array<number>|string|boolean>} variables Style variables
 * @property {boolean} [disableHitDetection=false] Setting this to true will provide a slight performance boost, but will
 * prevent all hit detection on the layer.
 * @property {number|'auto'} [textRenderThrottleMs='auto'] Minimum interval in ms between text overlay updates
 * (text render + instructions rebuild) when animating points. Use `'auto'` to adapt to device performance.
 * Use `0` to disable throttling (update on every change).
 * @property {Array<import("./Layer").PostProcessesOptions>} [postProcesses] Post-processes definitions
 */

/**
 * @classdesc
 * Experimental WebGL vector renderer. Supports polygons, lines and points:
 * Polygons are broken down into triangles
 * Lines are rendered as strips of quads
 * Points are rendered as quads
 *
 * You need to provide vertex and fragment shaders as well as custom attributes for each type of geometry. All shaders
 * can access the uniforms in the {@link module:ol/webgl/Helper~DefaultUniform} enum.
 * The vertex shaders can access the following attributes depending on the geometry type:
 * For polygons: {@link module:ol/render/webgl/PolygonBatchRenderer~Attributes}
 * For line strings: {@link module:ol/render/webgl/LineStringBatchRenderer~Attributes}
 * For points: {@link module:ol/render/webgl/PointBatchRenderer~Attributes}
 *
 * Please note that the fragment shaders output should have premultiplied alpha, otherwise visual anomalies may occur.
 *
 * Note: this uses {@link module:ol/webgl/Helper~WebGLHelper} internally.
 */
class WebGLVectorLayerRenderer extends WebGLLayerRenderer {
  /**
   * @param {import("../../layer/Layer.js").default} layer Layer.
   * @param {Options} options Options.
   */
  constructor(layer, options) {
    const uniforms = {
      [Uniforms.RENDER_EXTENT]: [0, 0, 0, 0],
      [Uniforms.PATTERN_ORIGIN]: [0, 0],
      [Uniforms.GLOBAL_ALPHA]: 1,
    };

    super(layer, {
      uniforms: uniforms,
      postProcesses: [
        createPostProcessDefinition(
          () => this.styleRenderer_.getTextOverlayCanvas(),
          () => this.styleRenderer_.getTextOverlayFrameState(),
        ),
        ...(options.postProcesses ?? []),
      ],
    });

    /**
     * last time of rederFrame call for FPS detection
     * @type {number}
     * @private
     */
    this.lastFrameTime_ = nowMs();

    /**
     * @type {boolean}
     * @private
     */
    this.hitDetectionEnabled_ = !options.disableHitDetection;

    /**
     * @type {WebGLRenderTarget}
     * @private
     */
    this.hitRenderTarget_;

    /**
     * @private
     */
    this.sourceRevision_ = -1;

    /**
     * @private
     */
    this.previousExtent_ = createEmpty();

    /**
     * @private
     */
    this.tmpCoords_ = [0, 0];
    /**
     * @private
     */
    this.tmpTransform_ = createTransform();
    /**
     * @private
     */
    this.tmpMat4_ = createMat4();

    /**
     * @type {import("../../transform.js").Transform}
     * @private
     */
    this.currentFrameStateTransform_ = createTransform();

    /**
     * Transform that was used when generating the current buffers
     * (world space -> clip space, in the [-1, 1] range).
     * This is used for fast point position updates (bufferSubData) without forcing
     * a full buffer rebuild.
     * @type {import("../../transform.js").Transform}
     * @private
     */
    this.renderTransform_ = createTransform();

    /**
     * Tracks the last known geometry revision per feature uid.
     * Used to detect "geometry-only" changes (e.g. Point coordinate animation)
     * and avoid expensive full buffer rebuilds when possible.
     * @type {Map<string, number>}
     * @private
     */
    this.geometryRevisionByUid_ = new Map();

    /**
     * Uids of point features whose coordinates changed since the last render.
     * @type {Set<string>}
     * @private
     */
    this.dirtyPointUids_ = new Set();

    /**
     * Maps feature uid -> instance index (or indices for MultiPoint) in the point
     * instance attributes buffer.
     * @type {Map<string, number|Array<number>>}
     * @private
     */
    this.pointInstanceIndexByUid_ = new Map();

    /**
     * Stride (in floats) for one point instance in the point instance attributes buffer.
     * Layout: [x, y, ...customAttributes]
     * @type {number}
     * @private
     */
    this.pointInstanceStride_ = 0;

    /**
     * Minimum interval in ms between text overlay updates (render + instructions rebuild)
     * during point animations. A value of 0 keeps the previous behavior (update on every change).
     * @type {number}
     * @private
     */
    this.textRenderThrottleMs_ = 0;

    /**
     * Whether text throttling is automatically derived from device performance.
     * @type {boolean}
     * @private
     */
    this.textRenderThrottleAuto_ = true;

    /**
     * Exponential moving average of text rebuild duration (ms).
     * @type {number}
     * @private
     */
    this.textRebuildDurationAvgMs_ = 0;

    /**
     * Exponential moving average of text render duration (ms).
     * @type {number}
     * @private
     */
    this.textRenderDurationAvgMs_ = 0;

    /**
     * Whether a text instructions refresh is needed due to point animation updates.
     * @type {boolean}
     * @private
     */
    this.textRebuildNeeded_ = false;

    /**
     * Whether a text instructions refresh is currently in flight.
     * @type {boolean}
     * @private
     */
    this.textRebuildInFlight_ = false;

    /**
     * Whether another text rebuild was requested while one was in flight.
     * @type {boolean}
     * @private
     */
    this.textRebuildQueued_ = false;

    /**
     * Last time (ms) text instructions were rebuilt.
     * @type {number}
     * @private
     */
    this.lastTextRenderTime_ = -Infinity;

    /**
     * Last time (ms) the text overlay was rendered.
     * @type {number}
     * @private
     */
    this.lastTextOverlayRenderTime_ = -Infinity;

    /**
     * Monotonic counter to discard outdated text rebuild results.
     * @type {number}
     * @private
     */
    this.textRebuildGeneration_ = 0;

    /**
     * Timeout id for delayed text rebuild scheduling.
     * @type {number}
     * @private
     */
    this.textRebuildTimerId_ = 0;

    /**
     * @type {import('../../style/flat.js').StyleVariables}
     * @private
     */
    this.styleVariables_ = {};

    /**
     * @type {LayerStyle}
     * @private
     */
    this.style_ = [];

    /**
     * @type {VectorStyleRenderer}
     * @public
     */
    this.styleRenderer_ = null;

    /**
     * @type {import('../../render/webgl/VectorStyleRenderer.js').WebGLBuffers}
     * @private
     */
    this.buffers_ = null;

    /**
     * @private
     * @type {number}
     */
    this.bufferGeneration_ = 0;

    /**
     * @private
     * @type {number}
     */
    this.bufferGenerationInFlight_ = 0;

    /**
     * @private
     * @type {boolean}
     */
    this.rebuildNeeded_ = false;

    /**
     * @private
     * @type {boolean}
     */
    this.rebuildQueued_ = false;

    /**
     * @private
     * @type {Array<string>}
     */
    this.pendingTextInstructions_ = [];

    /**
     * @private
     */
    this.batch_ = new MixedGeometryBatch();

    /**
     * @private
     * @type {boolean}
     */
    this.initialFeaturesAdded_ = false;

    /**
     * @private
     * @type {Array<import("../../events.js").EventsKey|null>}
     */
    this.sourceListenKeys_ = null;

    this.applyOptions_(options);
  }

  /**
   * @private
   * @param {import("../../Map.js").FrameState} frameState Frame state.
   */
  addInitialFeatures_(frameState) {
    const source = this.getLayer().getSource();
    const userProjection = getUserProjection();
    let projectionTransform;
    if (userProjection) {
      projectionTransform = getTransformFromProjections(
        userProjection,
        frameState.viewState.projection,
      );
    }
    const features = source.getFeatures();
    this.batch_.addFeatures(features, projectionTransform);
    // Seed geometry revisions so we can distinguish geometry-only changes later.
    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      const geometry = feature.getGeometry?.();
      if (geometry) {
        this.geometryRevisionByUid_.set(
          getUid(feature),
          geometry.getRevision(),
        );
      }
    }
    this.sourceListenKeys_ = [
      listen(
        source,
        VectorEventType.ADDFEATURE,
        this.handleSourceFeatureAdded_.bind(this, projectionTransform),
      ),
      listen(
        source,
        VectorEventType.CHANGEFEATURE,
        this.handleSourceFeatureChanged_.bind(this, projectionTransform),
        this,
      ),
      listen(
        source,
        VectorEventType.REMOVEFEATURE,
        this.handleSourceFeatureDelete_,
        this,
      ),
      listen(
        source,
        VectorEventType.CLEAR,
        this.handleSourceFeatureClear_,
        this,
      ),
    ];
  }

  /**
   * @param {Options} options Options.
   * @private
   */
  applyOptions_(options) {
    this.styleVariables_ = options.variables;
    this.style_ = options.style;
    const throttle = options.textRenderThrottleMs;
    if (throttle === undefined || throttle === null || throttle === 'auto') {
      this.textRenderThrottleAuto_ = true;
      this.textRenderThrottleMs_ = DEFAULT_TEXT_RENDER_THROTTLE_MS;
    } else {
      this.textRenderThrottleAuto_ = false;
      this.textRenderThrottleMs_ = Math.max(0, throttle);
    }
    this.textRebuildNeeded_ = false;
    this.textRebuildQueued_ = false;
    this.textRebuildInFlight_ = false;
    this.lastTextRenderTime_ = -Infinity;
    this.lastTextOverlayRenderTime_ = -Infinity;
    this.textRebuildDurationAvgMs_ = 0;
    this.textRenderDurationAvgMs_ = 0;
    if (this.textRebuildTimerId_) {
      clearTimeout(this.textRebuildTimerId_);
      this.textRebuildTimerId_ = 0;
    }
  }

  /**
   * @private
   */
  createRenderers_() {
    this.buffers_ = null;
    this.styleRenderer_ = new VectorStyleRenderer(
      this.style_,
      this.styleVariables_,
      this.helper,
      this.hitDetectionEnabled_,
    );
  }

  /**
   * @override
   */
  reset(options) {
    this.applyOptions_(options);
    if (this.helper) {
      this.createRenderers_();
    }
    super.reset(options);
  }

  /**
   * @override
   */
  afterHelperCreated() {
    if (this.styleRenderer_) {
      // To reuse buffers
      this.styleRenderer_.setHelper(this.helper, this.buffers_);
    } else {
      this.createRenderers_();
    }

    if (this.hitDetectionEnabled_) {
      this.hitRenderTarget_ = new WebGLRenderTarget(this.helper);
    }
  }

  /**
   * @param {import("../../proj.js").TransformFunction} projectionTransform Transform function.
   * @param {import("../../source/Vector.js").VectorSourceEvent} event Event.
   * @private
   */
  handleSourceFeatureAdded_(projectionTransform, event) {
    const feature = event.feature;
    this.batch_.addFeature(feature, projectionTransform);
    const geometry = feature.getGeometry?.();
    if (geometry) {
      this.geometryRevisionByUid_.set(getUid(feature), geometry.getRevision());
    }
    // A new feature changes buffers size/layout -> full rebuild.
    this.requestRebuild_();
  }

  /**
   * @param {import("../../proj.js").TransformFunction} projectionTransform Transform function.
   * @param {import("../../source/Vector.js").VectorSourceEvent} event Event.
   * @private
   */
  handleSourceFeatureChanged_(projectionTransform, event) {
    const feature = event.feature;
    const uid = getUid(feature);
    const geometry = feature.getGeometry?.();
    const geometryRevision = geometry ? geometry.getRevision() : -1;
    const previousRevision = this.geometryRevisionByUid_.get(uid);
    const geometryChanged =
      previousRevision === undefined || previousRevision !== geometryRevision;

    if (geometryChanged) {
      this.geometryRevisionByUid_.set(uid, geometryRevision);
    }

    // Fast path for animating points (coordinate updates): update instance buffer
    // positions instead of forcing a full rebuild.
    // Note: this is only safe when no user projection is used (no projectionTransform).
    // For text overlays, this fast path is only enabled when a text throttle is configured.
    if (
      geometryChanged &&
      !projectionTransform &&
      geometry &&
      geometry.getType() === 'Point' &&
      this.buffers_?.pointBuffers &&
      (!this.buffers_.textInstructionsKey || this.textRenderThrottleMs_ > 0) &&
      this.pointInstanceStride_ > 0 &&
      this.pointInstanceIndexByUid_.has(uid)
    ) {
      const pointGeometry =
        /** @type {import("../../geom/Point.js").default} */ (geometry);
      const batchEntry = this.batch_.pointBatch.entries[uid];
      // If the geometry object was replaced, the batch entry may reference stale coordinates.
      // In that case we fall back to a full batch update + rebuild.
      if (batchEntry?.flatCoordss?.[0] !== pointGeometry.getFlatCoordinates()) {
        // pass through to fallback below
      } else {
        // MixedGeometryBatch stores a reference to the Point's flatCoordinates array,
        // so we don't need to call `changeFeature()` for coordinate-only animations.
        this.dirtyPointUids_.add(uid);
        if (
          this.buffers_?.textInstructionsKey &&
          this.textRenderThrottleMs_ > 0
        ) {
          this.textRebuildNeeded_ = true;
        }
        return;
      }
    }

    // Fallback: keep the batch in sync and request a full rebuild.
    // This covers:
    // - property changes impacting style custom attributes
    // - geometry type changes / geometry replaced
    // - MultiPoint (coordinates are copied into the batch)
    // - user projection usage
    this.batch_.changeFeature(feature, projectionTransform);
    this.requestRebuild_(false);
  }

  /**
   * @param {import("../../source/Vector.js").VectorSourceEvent} event Event.
   * @private
   */
  handleSourceFeatureDelete_(event) {
    const feature = event.feature;
    this.batch_.removeFeature(feature);
    const uid = getUid(feature);
    this.geometryRevisionByUid_.delete(uid);
    this.dirtyPointUids_.delete(uid);
    this.pointInstanceIndexByUid_.delete(uid);
    // Feature removal changes buffers size/layout -> full rebuild.
    this.requestRebuild_();
  }

  /**
   * @private
   */
  handleSourceFeatureClear_() {
    this.batch_.clear();
    this.geometryRevisionByUid_.clear();
    this.dirtyPointUids_.clear();
    this.pointInstanceIndexByUid_.clear();
    this.pointInstanceStride_ = 0;
    this.textRebuildNeeded_ = false;
    this.textRebuildQueued_ = false;
    // Clearing changes buffers size/layout -> full rebuild.
    this.requestRebuild_();
  }

  /**
   * Request a full buffer rebuild. If a generation is already in flight,
   * it can be invalidated so outdated buffers are dropped when ready.
   * @param {boolean} [invalidateInFlight] Whether to invalidate in-flight generation.
   * @private
   */
  requestRebuild_(invalidateInFlight = true) {
    if (this.bufferGenerationInFlight_) {
      if (invalidateInFlight && !this.rebuildQueued_) {
        this.bufferGeneration_++;
      }
      this.rebuildQueued_ = true;
      this.rebuildNeeded_ = false;
      return;
    }
    this.rebuildNeeded_ = true;
  }

  /**
   * Rebuild the mapping between feature uids and point instance indices in the current
   * point instance attributes buffer.
   * @private
   */
  rebuildPointInstanceIndex_() {
    this.pointInstanceIndexByUid_.clear();
    this.pointInstanceStride_ = 0;
    if (!this.buffers_?.pointBuffers) {
      return;
    }

    const instanceAttributesBuffer = this.buffers_.pointBuffers[2];
    const pointCount = this.batch_.pointBatch.geometriesCount;
    if (!pointCount) {
      return;
    }

    const stride = instanceAttributesBuffer.getSize() / pointCount;
    // Guard against unexpected layouts.
    if (!Number.isFinite(stride) || !Number.isInteger(stride) || stride < 2) {
      return;
    }
    this.pointInstanceStride_ = stride;

    let instanceIndex = 0;
    const entries = this.batch_.pointBatch.entries;
    for (const uid in entries) {
      const entry = entries[uid];
      const geometriesCount = entry.flatCoordss.length;
      if (geometriesCount === 1) {
        this.pointInstanceIndexByUid_.set(uid, instanceIndex++);
        continue;
      }
      const indices = new Array(geometriesCount);
      for (let i = 0; i < geometriesCount; i++) {
        indices[i] = instanceIndex++;
      }
      this.pointInstanceIndexByUid_.set(uid, indices);
    }
  }

  /**
   * Apply pending point coordinate updates by updating the point instance attributes buffer.
   * This avoids full buffer rebuilds for animated points.
   * @param {WebGLRenderingContext} gl WebGL context.
   * @private
   */
  flushPointUpdates_(gl) {
    if (
      !this.dirtyPointUids_.size ||
      !this.buffers_?.pointBuffers ||
      !this.pointInstanceStride_ ||
      (this.buffers_.textInstructionsKey && this.textRenderThrottleMs_ === 0)
    ) {
      return;
    }

    const instanceAttributesBuffer = this.buffers_.pointBuffers[2];
    const array = /** @type {Float32Array|null} */ (
      instanceAttributesBuffer.getArray()
    );
    if (!array) {
      return;
    }

    // Project world coordinates to the coordinate system of the current buffers.
    const t = this.renderTransform_;

    let minInstance = Infinity;
    let maxInstance = -Infinity;
    const stride = this.pointInstanceStride_;

    for (const uid of this.dirtyPointUids_) {
      const entry = this.batch_.pointBatch.entries[uid];
      if (!entry) {
        continue;
      }
      const indices = this.pointInstanceIndexByUid_.get(uid);
      if (indices === undefined) {
        continue;
      }

      if (typeof indices === 'number') {
        const coords = entry.flatCoordss[0];
        const x = coords[0];
        const y = coords[1];
        const px = t[0] * x + t[2] * y + t[4];
        const py = t[1] * x + t[3] * y + t[5];
        const offset = indices * stride;
        array[offset] = px;
        array[offset + 1] = py;
        if (indices < minInstance) {
          minInstance = indices;
        }
        if (indices > maxInstance) {
          maxInstance = indices;
        }
        continue;
      }

      // MultiPoint: update all point instances for this feature.
      for (let i = 0; i < indices.length; i++) {
        const idx = indices[i];
        const coords = entry.flatCoordss[i];
        const x = coords[0];
        const y = coords[1];
        const px = t[0] * x + t[2] * y + t[4];
        const py = t[1] * x + t[3] * y + t[5];
        const offset = idx * stride;
        array[offset] = px;
        array[offset + 1] = py;
        if (idx < minInstance) {
          minInstance = idx;
        }
        if (idx > maxInstance) {
          maxInstance = idx;
        }
      }
    }

    this.dirtyPointUids_.clear();

    if (!Number.isFinite(minInstance) || !Number.isFinite(maxInstance)) {
      return;
    }

    // Upload a single contiguous subrange to keep GL calls minimal.
    const start = minInstance * stride;
    const end = (maxInstance + 1) * stride;
    this.helper.bindBuffer(instanceAttributesBuffer);
    gl.bufferSubData(
      instanceAttributesBuffer.getType(),
      start * Float32Array.BYTES_PER_ELEMENT,
      array.subarray(start, end),
    );
  }

  /**
   * Throttled refresh of text instructions for animated points.
   * @param {import("../../Map.js").FrameState} frameState Frame state.
   * @param {number} currentFps currentFps
   * @private
   */
  maybeRebuildTextInstructions_(frameState, currentFps) {
    if (
      !this.textRenderThrottleMs_ ||
      !this.textRebuildNeeded_ ||
      !this.styleRenderer_ ||
      !this.buffers_
    ) {
      return;
    }

    if (this.textRebuildInFlight_ || this.bufferGenerationInFlight_) {
      this.textRebuildQueued_ = true;
      return;
    }

    const now = frameState.time;
    if (now - this.lastTextRenderTime_ < this.textRenderThrottleMs_) {
      if (!this.textRebuildTimerId_) {
        const delay =
          this.textRenderThrottleMs_ - (now - this.lastTextRenderTime_);
        this.textRebuildTimerId_ = setTimeout(() => {
          this.textRebuildTimerId_ = 0;
          if (this.textRebuildNeeded_) {
            this.getLayer().changed();
          }
        }, delay);
      }
      return;
    }

    if (this.textRebuildTimerId_) {
      clearTimeout(this.textRebuildTimerId_);
      this.textRebuildTimerId_ = 0;
    }

    this.textRebuildInFlight_ = true;
    this.textRebuildNeeded_ = false;
    this.lastTextRenderTime_ = now;

    const generation = ++this.textRebuildGeneration_;
    const buffersRef = this.buffers_;
    const rebuildStart = nowMs();
    const transform = this.helper.makeProjectionTransform(
      frameState,
      createTransform(),
    );

    this.styleRenderer_
      .generateTextInstructionsOnly(this.batch_, transform)
      .then((textInstructionsKey) => {
        this.textRebuildInFlight_ = false;
        if (this.textRenderThrottleAuto_) {
          const duration = nowMs() - rebuildStart;
          this.textRebuildDurationAvgMs_ = this.textRebuildDurationAvgMs_
            ? this.textRebuildDurationAvgMs_ *
                (1 - TEXT_RENDER_THROTTLE_SMOOTHING) +
              duration * TEXT_RENDER_THROTTLE_SMOOTHING
            : duration;
          this.updateTextRenderThrottle_(currentFps);
        }

        if (!textInstructionsKey) {
          if (this.textRebuildQueued_) {
            this.textRebuildQueued_ = false;
            this.textRebuildNeeded_ = true;
          }
          return;
        }

        if (generation !== this.textRebuildGeneration_) {
          this.styleRenderer_.disposeTextInstructions(textInstructionsKey);
          return;
        }

        if (this.buffers_ !== buffersRef) {
          this.styleRenderer_.disposeTextInstructions(textInstructionsKey);
        } else {
          const previousKey = buffersRef.textInstructionsKey;
          buffersRef.textInstructionsKey = textInstructionsKey;
          this.queueTextInstructionsDispose_(previousKey);
          this.getLayer().changed();
        }

        if (this.textRebuildQueued_) {
          this.textRebuildQueued_ = false;
          this.textRebuildNeeded_ = true;
        }
      });
  }

  /**
   * @private
   */
  /**
   * Адаптивный расчет троттлинга на основе стоимости операции и текущего FPS.
   * @param {number} currentFps Мгновенный FPS карты (полезно для обнаружения просадок).
   * @private
   */
  updateTextRenderThrottle_(currentFps) {
    // 1. Оценка "стоимости" задачи (берем максимум из генерации инструкций или отрисовки)
    // Добавляем небольшой оверхед (10%), так как есть накладные расходы на postMessage и переключения контекста
    const taskCostMs =
      Math.max(this.textRebuildDurationAvgMs_, this.textRenderDurationAvgMs_) *
      1.1;

    // Если метрик пока нет, используем безопасный дефолт
    if (!taskCostMs) {
      this.textRenderThrottleMs_ = DEFAULT_TEXT_RENDER_THROTTLE_MS;
      return;
    }

    // 2. Определяем доступный бюджет (Duty Cycle)
    // Базовый бюджет - TARGET_DUTY_CYCLE.
    // Если FPS просел (карта лагает), мы уменьшаем бюджет пропорционально просадке.
    let effectiveDutyCycle = TARGET_DUTY_CYCLE;

    if (currentFps < CRITICAL_FPS_THRESHOLD) {
      // Пример: если FPS 30 при пороге 45, мы снижаем бюджет в (30/45) раз.
      // При FPS 30 бюджет станет 0.25 * 0.66 = 0.16 (16%)
      // При FPS 15 бюджет станет 0.25 * 0.33 = 0.08 (8%)
      const performanceFactor = Math.max(
        0.1,
        currentFps / CRITICAL_FPS_THRESHOLD,
      );
      effectiveDutyCycle *= performanceFactor;
    }

    // 3. Расчет идеального интервала
    // Формула: TotalTime = Cost / DutyCycle
    // Throttle (Wait) = TotalTime - Cost
    // Пример: Cost 10ms, Duty 0.2 (20%). Total = 50ms. Wait = 40ms.
    const idealInterval = taskCostMs / effectiveDutyCycle - taskCostMs;

    // 4. Сглаживание и ограничения
    // Используем простое линейное ограничение [MIN, MAX]
    const clampedInterval = Math.max(
      MIN_TEXT_RENDER_THROTTLE_MS,
      Math.min(MAX_TEXT_RENDER_THROTTLE_MS, idealInterval),
    );

    // Применяем сглаживание к самому значению троттлинга, чтобы не "скакало" слишком резко
    this.textRenderThrottleMs_ =
      this.textRenderThrottleMs_ * (1 - TEXT_RENDER_THROTTLE_SMOOTHING) +
      clampedInterval * TEXT_RENDER_THROTTLE_SMOOTHING;

    // (Опционально для дебага)
    // console.log(`Cost: ${taskCostMs.toFixed(1)}ms, FPS: ${currentFps.toFixed(0)}, Cycle: ${effectiveDutyCycle.toFixed(2)}, Throttle: ${this.textRenderThrottleMs_.toFixed(0)}ms`);
  }

  /**
   * Throttled render of the text overlay.
   * @param {import("../../Map.js").FrameState} frameState Frame state.
   * @param {number} currentFps current fps
   * @private
   */
  maybeFinalizeTextRender_(frameState, currentFps) {
    if (!this.styleRenderer_) {
      return;
    }

    if (this.batch_.isEmpty() || !this.styleRenderer_.hasText()) {
      this.styleRenderer_.clearTextOverlay();
      return;
    }

    if (!this.buffers_?.textInstructionsKey) {
      // If buffers are temporarily unavailable, keep the previous overlay
      // to avoid visible flicker during rebuilds.
      return;
    }

    const now = frameState.time;
    if (
      this.textRenderThrottleMs_ > 0 &&
      now - this.lastTextOverlayRenderTime_ < this.textRenderThrottleMs_
    ) {
      return;
    }
    this.lastTextOverlayRenderTime_ = now;

    // FIX: Flush pending instructions BEFORE starting the render.
    // This ensures that the worker receives the DISPOSE message for old keys
    // before the RENDER message, preventing it from drawing both the old and new text
    // in the same frame (which caused ghosting and opacity accumulation).
    this.flushPendingTextInstructions_();

    const renderStart = nowMs();
    this.styleRenderer_.finalizeTextRender(frameState).then(() => {
      // Removed flushPendingTextInstructions_() from here
      if (this.textRenderThrottleAuto_) {
        const duration = nowMs() - renderStart;
        this.textRenderDurationAvgMs_ = this.textRenderDurationAvgMs_
          ? this.textRenderDurationAvgMs_ *
              (1 - TEXT_RENDER_THROTTLE_SMOOTHING) +
            duration * TEXT_RENDER_THROTTLE_SMOOTHING
          : duration;
        this.updateTextRenderThrottle_(currentFps);
      }
    });
  }

  /**
   * @param {import("../../transform.js").Transform} batchInvertTransform Inverse of the transformation in which geometries are expressed
   * @private
   */
  applyUniforms_(batchInvertTransform) {
    // world to screen matrix
    setFromTransform(this.tmpTransform_, this.currentFrameStateTransform_);
    multiplyTransform(this.tmpTransform_, batchInvertTransform);
    this.helper.setUniformMatrixValue(
      Uniforms.PROJECTION_MATRIX,
      mat4FromTransform(this.tmpMat4_, this.tmpTransform_),
    );

    // screen to world matrix
    makeInverseTransform(this.tmpTransform_, this.tmpTransform_);
    this.helper.setUniformMatrixValue(
      Uniforms.SCREEN_TO_WORLD_MATRIX,
      mat4FromTransform(this.tmpMat4_, this.tmpTransform_),
    );

    // pattern origin should always be [0, 0] in world coordinates
    this.tmpCoords_[0] = 0;
    this.tmpCoords_[1] = 0;
    makeInverseTransform(this.tmpTransform_, batchInvertTransform);
    applyTransform(this.tmpTransform_, this.tmpCoords_);
    this.helper.setUniformFloatVec2(Uniforms.PATTERN_ORIGIN, this.tmpCoords_);
  }

  /**
   * Render the layer.
   * @param {import("../../Map.js").FrameState} frameState Frame state.
   * @return {HTMLElement} The rendered element.
   * @override
   */
  renderFrame(frameState) {
    const now = nowMs();
    const frameDuration = now - this.lastFrameTime_;
    this.lastFrameTime_ = now;

    // Protection against zero-case and anomalies (minimum 1 frame)
    const currentFps =
      frameDuration > 0 ? 1000 / Math.max(frameDuration, 16.6) : 60;

    const gl = this.helper.getGL();
    this.preRender(gl, frameState);

    const [startWorld, endWorld, worldWidth] = getWorldParameters(
      frameState,
      this.getLayer(),
    );

    // Apply any pending fast position updates before drawing.
    this.flushPointUpdates_(gl);
    // Throttled text refresh during animations (if enabled).
    this.maybeRebuildTextInstructions_(frameState, currentFps);

    // draw the normal canvas
    this.helper.prepareDraw(frameState);
    this.renderWorlds(frameState, false, startWorld, endWorld, worldWidth);

    this.maybeFinalizeTextRender_(frameState, currentFps);

    this.helper.finalizeDraw(
      frameState,
      this.dispatchPreComposeEvent,
      this.dispatchPostComposeEvent,
    );

    const canvas = this.helper.getCanvas();

    if (this.hitDetectionEnabled_) {
      this.renderWorlds(frameState, true, startWorld, endWorld, worldWidth);
      this.hitRenderTarget_.clearCachedData();
    }

    this.postRender(gl, frameState);

    return canvas;
  }

  /**
   * Determine whether renderFrame should be called.
   * @param {import("../../Map.js").FrameState} frameState Frame state.
   * @return {boolean} Layer is ready to be rendered.
   * @override
   */
  prepareFrameInternal(frameState) {
    if (!this.initialFeaturesAdded_) {
      this.addInitialFeatures_(frameState);
      this.initialFeaturesAdded_ = true;
    }

    const layer = this.getLayer();
    const vectorSource = layer.getSource();
    const viewState = frameState.viewState;
    const viewNotMoving =
      !frameState.viewHints[ViewHint.ANIMATING] &&
      !frameState.viewHints[ViewHint.INTERACTING];
    const extentChanged = !equals(this.previousExtent_, frameState.extent);
    const needsRebuild =
      viewNotMoving &&
      (extentChanged || this.rebuildNeeded_ || this.rebuildQueued_);

    if (needsRebuild) {
      if (this.bufferGenerationInFlight_) {
        if (!this.rebuildQueued_) {
          // Invalidate the in-flight generation so outdated buffers are dropped.
          this.bufferGeneration_++;
          this.rebuildQueued_ = true;
        }
        // coalesce rebuild requests while a generation is in flight
        this.rebuildNeeded_ = false;
        return true;
      }

      this.rebuildQueued_ = false;
      this.rebuildNeeded_ = false;

      const projection = viewState.projection;
      const resolution = viewState.resolution;

      const renderBuffer =
        layer instanceof BaseVector ? layer.getRenderBuffer() : 0;
      const extent = buffer(frameState.extent, renderBuffer * resolution);

      const userProjection = getUserProjection();
      if (userProjection) {
        vectorSource.loadFeatures(
          toUserExtent(extent, userProjection),
          toUserResolution(resolution, projection),
          userProjection,
        );
      } else {
        vectorSource.loadFeatures(extent, resolution, projection);
      }

      this.ready = false;

      const generation = ++this.bufferGeneration_;
      this.bufferGenerationInFlight_ = generation;

      const transform = this.helper.makeProjectionTransform(
        frameState,
        createTransform(),
      );

      this.styleRenderer_
        .generateBuffers(this.batch_, transform)
        .then((buffers) => {
          if (generation !== this.bufferGeneration_) {
            if (buffers) {
              this.disposeBuffers(buffers);
            }
            if (this.bufferGenerationInFlight_ === generation) {
              this.bufferGenerationInFlight_ = 0;
            }
            if (this.rebuildQueued_) {
              this.getLayer().changed();
            }
            return;
          }

          this.bufferGenerationInFlight_ = 0;

          if (this.buffers_) {
            this.disposeBuffers(this.buffers_);
          }
          this.buffers_ = buffers;
          // New buffers include fresh text instructions; clear pending text refreshes.
          this.textRebuildGeneration_++;
          this.textRebuildNeeded_ = false;
          this.textRebuildQueued_ = false;
          this.textRebuildInFlight_ = false;
          if (this.textRebuildTimerId_) {
            clearTimeout(this.textRebuildTimerId_);
            this.textRebuildTimerId_ = 0;
          }
          // Keep the transform used when generating buffers so we can update point positions
          // in the same coordinate system without rebuilding all buffers.
          setFromTransform(this.renderTransform_, transform);
          this.rebuildPointInstanceIndex_();
          this.ready = true;
          this.getLayer().changed();

          if (this.rebuildQueued_) {
            this.rebuildQueued_ = false;
            this.getLayer().changed();
          }
        });

      this.previousExtent_ = frameState.extent.slice();
    }

    return true;
  }

  /**
   * Render the world, either to the main framebuffer or to the hit framebuffer
   * @param {import("../../Map.js").FrameState} frameState current frame state
   * @param {boolean} forHitDetection whether the rendering is for hit detection
   * @param {number} startWorld the world to render in the first iteration
   * @param {number} endWorld the last world to render
   * @param {number} worldWidth the width of the worlds being rendered
   */
  renderWorlds(frameState, forHitDetection, startWorld, endWorld, worldWidth) {
    let world = startWorld;

    if (forHitDetection) {
      this.hitRenderTarget_.setSize([
        Math.floor(frameState.size[0] / 2),
        Math.floor(frameState.size[1] / 2),
      ]);
      this.helper.prepareDrawToRenderTarget(
        frameState,
        this.hitRenderTarget_,
        true,
      );
    }

    do {
      this.helper.makeProjectionTransform(
        frameState,
        this.currentFrameStateTransform_,
      );
      translateTransform(
        this.currentFrameStateTransform_,
        world * worldWidth,
        0,
      );
      if (!this.buffers_) {
        continue;
      }
      this.styleRenderer_.render(this.buffers_, frameState, () => {
        this.applyUniforms_(this.buffers_.invertVerticesTransform);
        this.helper.applyHitDetectionUniform(forHitDetection);
      });
    } while (++world < endWorld);
  }

  /**
   * Queue text instructions for disposal after the current text render completes.
   * @param {string|null|undefined} key Text instructions key.
   * @private
   */
  queueTextInstructionsDispose_(key) {
    if (!key) {
      return;
    }
    this.pendingTextInstructions_.push(key);
  }

  /**
   * Flush queued text instruction disposals.
   * @private
   */
  flushPendingTextInstructions_() {
    if (!this.pendingTextInstructions_.length || !this.styleRenderer_) {
      return;
    }
    for (const key of this.pendingTextInstructions_) {
      this.styleRenderer_.disposeTextInstructions(key);
    }
    this.pendingTextInstructions_.length = 0;
  }

  /**
   * @param {import("../../coordinate.js").Coordinate} coordinate Coordinate.
   * @param {import("../../Map.js").FrameState} frameState Frame state.
   * @param {number} hitTolerance Hit tolerance in pixels.
   * @param {import("../vector.js").FeatureCallback<T>} callback Feature callback.
   * @param {Array<import("../Map.js").HitMatch<T>>} matches The hit detected matches with tolerance.
   * @return {T|undefined} Callback result.
   * @template T
   * @override
   */
  forEachFeatureAtCoordinate(
    coordinate,
    frameState,
    hitTolerance,
    callback,
    matches,
  ) {
    assert(
      this.hitDetectionEnabled_,
      '`forEachFeatureAtCoordinate` cannot be used on a WebGL layer if the hit detection logic has been disabled using the `disableHitDetection: true` option.',
    );
    if (!this.styleRenderer_ || !this.hitDetectionEnabled_) {
      return undefined;
    }

    const pixel = applyTransform(
      frameState.coordinateToPixelTransform,
      coordinate.slice(),
    );

    const data = this.hitRenderTarget_.readPixel(pixel[0] / 2, pixel[1] / 2);
    const color = [data[0] / 255, data[1] / 255, data[2] / 255, data[3] / 255];
    const ref = colorDecodeId(color);
    const feature = this.batch_.getFeatureFromRef(ref);
    if (feature) {
      return callback(feature, this.getLayer(), null);
    }
    return undefined;
  }

  /**
   * Will release a set of Webgl buffers
   * @param {import('../../render/webgl/VectorStyleRenderer.js').WebGLBuffers} buffers Buffers
   */
  disposeBuffers(buffers) {
    /**
     * @param {Array<import('../../webgl/Buffer.js').default>} typeBuffers Buffers
     */
    const disposeBuffersOfType = (typeBuffers) => {
      for (const buffer of typeBuffers) {
        if (buffer) {
          this.helper.deleteBuffer(buffer);
        }
      }
    };
    if (buffers.pointBuffers) {
      disposeBuffersOfType(buffers.pointBuffers);
    }
    if (buffers.lineStringBuffers) {
      disposeBuffersOfType(buffers.lineStringBuffers);
    }
    if (buffers.polygonBuffers) {
      disposeBuffersOfType(buffers.polygonBuffers);
    }
    this.queueTextInstructionsDispose_(buffers.textInstructionsKey);
  }

  /**
   * Clean up.
   * @override
   */
  disposeInternal() {
    if (this.buffers_) {
      this.disposeBuffers(this.buffers_);
    }
    this.flushPendingTextInstructions_();
    if (this.textRebuildTimerId_) {
      clearTimeout(this.textRebuildTimerId_);
      this.textRebuildTimerId_ = 0;
    }
    if (this.sourceListenKeys_) {
      this.sourceListenKeys_.forEach(function (key) {
        unlistenByKey(key);
      });
      this.sourceListenKeys_ = null;
    }
    if (this.styleRenderer_) {
      this.styleRenderer_.dispose();
    }
    super.disposeInternal();
  }

  renderDeclutter() {}
}

export default WebGLVectorLayerRenderer;

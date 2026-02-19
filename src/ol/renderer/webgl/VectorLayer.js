/**
 * @module ol/renderer/webgl/VectorLayer
 */
import ViewHint from '../../ViewHint.js';
import {assert} from '../../asserts.js';
import {asString as colorAsString} from '../../color.js';
import {createCanvasContext2D} from '../../dom.js';
import {listen, unlistenByKey} from '../../events.js';
import {newParsingContext} from '../../expr/expression.js';
import {buffer, createEmpty, equals} from '../../extent.js';
import BaseVector from '../../layer/BaseVector.js';
import {
  getTransformFromProjections,
  getUserProjection,
  toUserExtent,
  toUserResolution,
} from '../../proj.js';
import {flatStyleLikeToStyleFunction} from '../../render/canvas/style.js';
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

const DEFAULT_TEXT_REBUILD_THROTTLE_MS = 120;
const DEFAULT_TEXT_OVERLAY_RENDER_THROTTLE_MS = 48;
const MIN_TEXT_RENDER_THROTTLE_MS = 16;
const MAX_TEXT_REBUILD_THROTTLE_MS = 2000;
const MAX_TEXT_OVERLAY_RENDER_THROTTLE_MS = 400;
const MAX_TEXT_OVERLAY_RENDER_THROTTLE_MOVING_MS = 80;
const TARGET_REBUILD_DUTY_CYCLE = 0.22;
const TARGET_OVERLAY_RENDER_DUTY_CYCLE = 0.35;
const CRITICAL_FPS_THRESHOLD = 45;
const TEXT_RENDER_THROTTLE_SMOOTHING = 0.2;
const TEXT_RENDER_FPS_SMOOTHING = 0.2;
const TEXT_REBUILD_VISIBILITY_PADDING_CLIP = 0.2;
const GPU_STATIC_LABEL_MAX_ATLAS_SIZE = 8192;
const GPU_STATIC_LABEL_MIN_ATLAS_SIZE = 512;
const GPU_STATIC_LABEL_SHELF_PADDING = 1;
const GPU_STATIC_LABEL_ATLAS_PADDING = 2;
const GPU_STATIC_LABEL_LINE_HEIGHT_FACTOR = 1.2;
const GPU_STATIC_LABEL_DEFAULT_FONT = '10px sans-serif';
const GPU_STATIC_LABEL_REFRESH_IDLE_DELAY_MS = 120;
const GPU_GLYPH_MAX_LABEL_LENGTH = 24;
const GPU_GLYPH_BITMAP_PADDING = 1;
const GPU_GLYPH_SDF_RADIUS = 8;
const GPU_GLYPH_DEFAULT_FILL_COLOR = '#333';
const GPU_GLYPH_SDF_CUTOFF = 0.5;
const GPU_GLYPH_SDF_SMOOTHING_PX = 1.25;

function nowMs() {
  return typeof performance !== 'undefined' && performance.now
    ? performance.now()
    : Date.now();
}

/**
 * @param {unknown} value Value.
 * @return {boolean} Whether value is a non-null object.
 */
function isObject(value) {
  return value !== null && typeof value === 'object';
}

/**
 * @param {unknown} value Value.
 * @return {boolean} Whether value is a style shader entry.
 */
function isStyleShaderEntry(value) {
  return isObject(value) && 'builder' in value;
}

/**
 * @param {unknown} value Value.
 * @return {boolean} Whether value is a style rule entry.
 */
function isStyleRuleEntry(value) {
  return isObject(value) && 'style' in value;
}

/**
 * @param {Object<string, *>} style Flat style.
 * @return {boolean} Whether style contains text properties.
 */
function hasTextStyleProperties(style) {
  for (const key in style) {
    if (key.startsWith('text-')) {
      return true;
    }
  }
  return false;
}

/**
 * @param {Object<string, *>} style Flat style.
 * @return {Object<string, *>} Shallow clone without text properties.
 */
function cloneStyleWithoutTextProperties(style) {
  const clone = {};
  for (const key in style) {
    if (!key.startsWith('text-')) {
      clone[key] = style[key];
    }
  }
  return clone;
}

/**
 * @param {Object<string, *>} style Flat style.
 * @return {Object<string, *>} Shallow clone with text properties only.
 */
function cloneTextOnlyStyle(style) {
  const clone = {};
  for (const key in style) {
    if (key.startsWith('text-')) {
      clone[key] = style[key];
    }
  }
  return clone;
}

/**
 * @param {Object<string, *>} obj Object.
 * @return {number} Number of enumerable own properties.
 */
function countOwnProperties(obj) {
  let count = 0;
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      count++;
    }
  }
  return count;
}

/**
 * @param {Array<number>|null|undefined} padding Padding.
 * @return {Array<number>} Normalized [top, right, bottom, left] padding.
 */
function normalizePadding(padding) {
  if (!padding || padding.length !== 4) {
    return [0, 0, 0, 0];
  }
  return [padding[0] || 0, padding[1] || 0, padding[2] || 0, padding[3] || 0];
}

/**
 * @param {string|Array<string>|undefined} text Text value.
 * @return {string} Normalized text.
 */
function normalizeTextValue(text) {
  if (Array.isArray(text)) {
    return text.join('\n');
  }
  return typeof text === 'string' ? text : '';
}

/**
 * Normalize text for glyph-atlas mode: single line, bounded length.
 * @param {string|Array<string>|undefined} text Text value.
 * @return {string} Normalized text.
 */
function normalizeGlyphTextValue(text) {
  const normalized = normalizeTextValue(text)
    .replace(/\r/g, '')
    .replace(/\n/g, ' ');
  if (!normalized) {
    return '';
  }
  const chars = Array.from(normalized);
  return chars.slice(0, GPU_GLYPH_MAX_LABEL_LENGTH).join('');
}

/**
 * @param {string|undefined} font Font.
 * @return {number} Font size in px.
 */
function parseFontSizePx(font) {
  if (!font) {
    return 10;
  }
  const match = /([0-9]+(?:\.[0-9]+)?)px/.exec(font);
  if (!match) {
    return 10;
  }
  const size = Number(match[1]);
  return Number.isFinite(size) && size > 0 ? size : 10;
}

/**
 * @param {number} value Value.
 * @return {number} Next power of two.
 */
function nextPowerOfTwo(value) {
  if (value <= 1) {
    return 1;
  }
  return 2 ** Math.ceil(Math.log2(value));
}

const EDT_INF = 1e20;

/**
 * 1D squared distance transform (Felzenszwalb/Huttenlocher).
 * @param {Float32Array} f Input costs.
 * @param {number} n Length.
 * @param {Float32Array} d Output distances.
 * @param {Int32Array} v Temporary parabola locations.
 * @param {Float32Array} z Temporary parabola boundaries.
 */
function edt1d(f, n, d, v, z) {
  let k = 0;
  v[0] = 0;
  z[0] = -EDT_INF;
  z[1] = EDT_INF;

  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]) || 0;
    while (s <= z[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]) || 0;
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = EDT_INF;
  }

  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) {
      k++;
    }
    const diff = q - v[k];
    d[q] = diff * diff + f[v[k]];
  }
}

/**
 * Compute squared euclidean distance to nearest `target` binary pixel.
 * @param {Uint8Array} binaryMask Binary mask (0/1).
 * @param {number} width Width.
 * @param {number} height Height.
 * @param {number} target Target bit (0 or 1).
 * @return {Float32Array} Squared distance field.
 */
function computeSquaredDistanceTransform(binaryMask, width, height, target) {
  const total = width * height;
  const base = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    base[i] = binaryMask[i] === target ? 0 : EDT_INF;
  }

  const temp = new Float32Array(total);
  const maxLen = Math.max(width, height);
  const f = new Float32Array(maxLen);
  const d = new Float32Array(maxLen);
  const v = new Int32Array(maxLen);
  const z = new Float32Array(maxLen + 1);

  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x++) {
      f[x] = base[rowOffset + x];
    }
    edt1d(f, width, d, v, z);
    for (let x = 0; x < width; x++) {
      temp[rowOffset + x] = d[x];
    }
  }

  const output = new Float32Array(total);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      f[y] = temp[y * width + x];
    }
    edt1d(f, height, d, v, z);
    for (let y = 0; y < height; y++) {
      output[y * width + x] = d[y];
    }
  }

  return output;
}

/**
 * @param {string|undefined} align Align.
 * @return {number} Anchor x in fraction units.
 */
function textAlignToAnchorX(align) {
  switch (align) {
    case 'left':
    case 'start':
      return 0;
    case 'right':
    case 'end':
      return 1;
    default:
      return 0.5;
  }
}

/**
 * @param {string|undefined} baseline Baseline.
 * @return {number} Anchor y in fraction units.
 */
function textBaselineToAnchorY(baseline) {
  switch (baseline) {
    case 'top':
    case 'hanging':
      return 0;
    case 'bottom':
    case 'alphabetic':
    case 'ideographic':
      return 1;
    default:
      return 0.5;
  }
}

/**
 * @param {number|Array<*>|undefined} value Number expression.
 * @return {number|Array<*>} Negated expression.
 */
function negateNumberExpression(value) {
  if (typeof value === 'number') {
    return -value;
  }
  if (value === undefined) {
    return 0;
  }
  return ['*', -1, value];
}

/**
 * @param {number|Array<*>|undefined} offsetX X expression.
 * @param {number|Array<*>|undefined} offsetY Y expression.
 * @return {Array<number>|Array<*>|null} Icon displacement expression.
 */
function createTextDisplacementExpression(offsetX, offsetY) {
  const x = offsetX === undefined ? 0 : offsetX;
  const y = offsetY === undefined ? 0 : offsetY;
  if (typeof x === 'number' && typeof y === 'number') {
    if (x === 0 && y === 0) {
      return null;
    }
    return [x, -y];
  }
  return ['array', x, negateNumberExpression(y)];
}

/**
 * @param {*} color Color-like.
 * @return {string|CanvasPattern|CanvasGradient|null} Canvas compatible style.
 */
function toCanvasStyleColor(color) {
  if (typeof color === 'string') {
    return color;
  }
  if (Array.isArray(color)) {
    return colorAsString(color);
  }
  if (
    typeof ArrayBuffer !== 'undefined' &&
    color &&
    ArrayBuffer.isView(color) &&
    color.length >= 3
  ) {
    return colorAsString(Array.prototype.slice.call(color, 0, 4));
  }
  if (
    typeof CanvasGradient !== 'undefined' &&
    color instanceof CanvasGradient
  ) {
    return color;
  }
  if (typeof CanvasPattern !== 'undefined' && color instanceof CanvasPattern) {
    return color;
  }
  return null;
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
 * @property {boolean|'soft'|'strict'|'glyph-soft'|'glyph-strict'} [gpuStaticLabel=false] Enables radical text mode where point labels
 * are rendered as symbols in the same pass as icons. `true` is equivalent to `'strict'` (single initial bake string atlas).
 * Use `'soft'` to refresh the string atlas on source add/remove/property changes.
 * Use `'glyph-soft'` / `'glyph-strict'` for an experimental glyph-atlas path (memory scales with unique glyphs instead of unique labels).
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
     * Minimum interval in ms between text instruction rebuilds during point animations.
     * A value of 0 keeps the previous behavior (rebuild on every change).
     * @type {number}
     * @private
     */
    this.textRenderThrottleMs_ = 0;

    /**
     * Minimum interval in ms between text overlay render passes.
     * @type {number}
     * @private
     */
    this.textOverlayRenderThrottleMs_ = 0;

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
     * Exponential moving average of frame FPS used by adaptive throttling.
     * @type {number}
     * @private
     */
    this.currentFpsAvg_ = 60;

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
     * Last text instructions key successfully rendered to the text overlay.
     * Used to force an immediate overlay refresh when instructions were replaced.
     * @type {string|null}
     * @private
     */
    this.lastRenderedTextInstructionsKey_ = null;

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
     * Raw style provided by the layer options.
     * Used as source when building the static GPU label variant.
     * @type {LayerStyle}
     * @private
     */
    this.sourceStyle_ = [];

    /**
     * Enable radical text path: labels are baked into a static atlas and rendered as icons.
     * @type {boolean}
     * @private
     */
    this.gpuStaticLabelEnabled_ = false;

    /**
     * Static label operation mode: `off`, `strict`, or `soft`.
     * @type {'off'|'strict'|'soft'}
     * @private
     */
    this.gpuStaticLabelMode_ = 'off';

    /**
     * Whether static label atlas should be refreshed incrementally.
     * @type {boolean}
     * @private
     */
    this.gpuStaticLabelSoftMode_ = false;

    /**
     * Whether experimental glyph-atlas mode is enabled.
     * @type {boolean}
     * @private
     */
    this.gpuStaticLabelGlyphMode_ = false;

    /**
     * Whether the radical static-label style has been applied.
     * @type {boolean}
     * @private
     */
    this.gpuStaticLabelReady_ = false;

    /**
     * Whether static-label initialization has already been attempted for current options.
     * @type {boolean}
     * @private
     */
    this.gpuStaticLabelInitAttempted_ = false;

    /**
     * Incremental id used to generate stable feature property names for static label atlases.
     * @type {number}
     * @private
     */
    this.gpuStaticLabelStyleId_ = 0;

    /**
     * Whether a soft static label atlas refresh is pending.
     * @type {boolean}
     * @private
     */
    this.gpuStaticLabelRefreshPending_ = false;

    /**
     * Last timestamp (ms) when a point geometry animation update was observed.
     * Used to postpone soft atlas refresh while animation is active.
     * @type {number}
     * @private
     */
    this.lastPointAnimationUpdateTime_ = -Infinity;

    /**
     * Static label feature property names written during atlas generation.
     * @type {Array<{offsetXProperty?: string, offsetYProperty?: string, widthProperty?: string, heightProperty?: string, properties?: Array<string>}>}
     * @private
     */
    this.gpuStaticLabelProperties_ = [];

    /**
     * 2D context reused to measure and draw static text atlases.
     * @type {CanvasRenderingContext2D}
     * @private
     */
    this.gpuStaticLabelContext_ = createCanvasContext2D(1, 1);

    /**
     * 2D context used to rasterize one glyph mask before SDF conversion.
     * @type {CanvasRenderingContext2D}
     * @private
     */
    this.gpuStaticLabelGlyphRasterContext_ = createCanvasContext2D(1, 1);

    /**
     * Runtime atlas size cap derived from WebGL MAX_TEXTURE_SIZE.
     * @type {number}
     * @private
     */
    this.gpuStaticLabelMaxAtlasSize_ = GPU_STATIC_LABEL_MAX_ATLAS_SIZE;

    /**
     * Atlas canvases currently referenced by static-label icon styles.
     * Stored so we can aggressively release old atlas memory on refresh.
     * @type {Array<HTMLCanvasElement|OffscreenCanvas>}
     * @private
     */
    this.gpuStaticLabelAtlases_ = [];

    /**
     * Reusable 1x1 white canvas used for background quads in glyph mode.
     * @type {HTMLCanvasElement|OffscreenCanvas|null}
     * @private
     */
    this.gpuStaticLabelWhitePixelAtlas_ = null;

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
    this.maybeInitializeGpuStaticLabels_(features);
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
   * Initialize static GPU label mode once initial features are available.
   * @param {Array<import('../../Feature.js').FeatureLike>} features Features.
   * @private
   */
  maybeInitializeGpuStaticLabels_(features) {
    if (!this.gpuStaticLabelEnabled_ || this.gpuStaticLabelInitAttempted_) {
      return;
    }
    this.gpuStaticLabelInitAttempted_ = true;
    this.releaseGpuStaticLabelAtlases_();
    this.gpuStaticLabelProperties_.length = 0;
    const transformedStyle = this.transformStyleForGpuStaticLabels_(
      this.sourceStyle_,
      features,
    );
    if (!transformedStyle) {
      this.style_ = this.sourceStyle_;
      this.gpuStaticLabelReady_ = false;
      return;
    }

    this.style_ = transformedStyle;
    this.gpuStaticLabelReady_ = true;

    if (this.helper) {
      this.createRenderers_();
      this.requestRebuild_();
    }
  }

  /**
   * Queue a soft static-label atlas refresh.
   * @private
   */
  queueGpuStaticLabelRefresh_() {
    if (!this.gpuStaticLabelReady_ || !this.gpuStaticLabelSoftMode_) {
      return;
    }
    if (!this.gpuStaticLabelRefreshPending_) {
      this.gpuStaticLabelRefreshPending_ = true;
      this.getLayer().changed();
    }
  }

  /**
   * Refresh static-label atlas and style from current source features.
   * @return {boolean} Whether refresh succeeded and static mode stays enabled.
   * @private
   */
  refreshGpuStaticLabels_() {
    if (!this.gpuStaticLabelEnabled_) {
      return false;
    }
    const source = this.getLayer().getSource();
    const features = source.getFeatures();
    this.releaseGpuStaticLabelAtlases_();
    this.gpuStaticLabelProperties_.length = 0;
    this.gpuStaticLabelStyleId_ = 0;
    const transformedStyle = this.transformStyleForGpuStaticLabels_(
      this.sourceStyle_,
      features,
    );
    if (!transformedStyle) {
      this.style_ = this.sourceStyle_;
      this.gpuStaticLabelReady_ = false;
      this.gpuStaticLabelRefreshPending_ = false;
      this.createRenderers_();
      this.requestRebuild_();
      return false;
    }
    this.style_ = transformedStyle;
    this.gpuStaticLabelReady_ = true;
    this.gpuStaticLabelRefreshPending_ = false;
    this.createRenderers_();
    this.requestRebuild_();
    return true;
  }

  /**
   * Release old static-label atlas canvases to keep memory bounded after refreshes.
   * @private
   */
  releaseGpuStaticLabelAtlases_() {
    for (let i = 0; i < this.gpuStaticLabelAtlases_.length; i++) {
      const canvas = this.gpuStaticLabelAtlases_[i];
      canvas.width = 1;
      canvas.height = 1;
    }
    this.gpuStaticLabelAtlases_.length = 0;
    this.gpuStaticLabelWhitePixelAtlas_ = null;
    this.gpuStaticLabelGlyphRasterContext_.canvas.width = 1;
    this.gpuStaticLabelGlyphRasterContext_.canvas.height = 1;
  }

  /**
   * @return {boolean} Whether point geometry animation updates are still active.
   * @private
   */
  isPointAnimationActive_() {
    return (
      nowMs() - this.lastPointAnimationUpdateTime_ <
      GPU_STATIC_LABEL_REFRESH_IDLE_DELAY_MS
    );
  }

  /**
   * Initialize static-label feature properties with empty values.
   * @param {import('../../Feature.js').FeatureLike} feature Feature.
   * @private
   */
  initializeStaticLabelFeatureProperties_(feature) {
    if (typeof feature.set !== 'function') {
      return;
    }
    for (let i = 0; i < this.gpuStaticLabelProperties_.length; i++) {
      const names = this.gpuStaticLabelProperties_[i];
      if (Array.isArray(names.properties)) {
        for (let j = 0; j < names.properties.length; j++) {
          feature.set(names.properties[j], 0, true);
        }
        continue;
      }
      if (names.offsetXProperty) {
        feature.set(names.offsetXProperty, 0, true);
      }
      if (names.offsetYProperty) {
        feature.set(names.offsetYProperty, 0, true);
      }
      if (names.widthProperty) {
        feature.set(names.widthProperty, 0, true);
      }
      if (names.heightProperty) {
        feature.set(names.heightProperty, 0, true);
      }
    }
  }

  /**
   * Transform layer style into a static label icon variant.
   * Returns null when style cannot be transformed safely.
   * @param {LayerStyle} style Layer style.
   * @param {Array<import('../../Feature.js').FeatureLike>} features Features.
   * @return {LayerStyle|null} Transformed style.
   * @private
   */
  transformStyleForGpuStaticLabels_(style, features) {
    if (!style || !isObject(style)) {
      return null;
    }

    if (Array.isArray(style)) {
      if (!style.length) {
        return style;
      }

      if (isStyleShaderEntry(style[0])) {
        return null;
      }

      if (isStyleRuleEntry(style[0])) {
        const rules = /** @type {Array<import('../../style/flat.js').Rule>} */ (
          style
        );
        const transformedRules = new Array(rules.length);
        for (let i = 0; i < rules.length; i++) {
          const rule = rules[i];
          const transformedRuleStyle = this.transformStyleForGpuStaticLabels_(
            rule.style,
            features,
          );
          if (!transformedRuleStyle) {
            return null;
          }
          transformedRules[i] = {
            ...rule,
            style: transformedRuleStyle,
          };
        }
        return transformedRules;
      }

      const flatStyles =
        /** @type {Array<import('../../style/flat.js').FlatStyle>} */ (style);
      const transformedStyles = [];
      for (let i = 0; i < flatStyles.length; i++) {
        const transformed = this.transformFlatStyleForGpuStaticLabels_(
          flatStyles[i],
          features,
        );
        if (!transformed) {
          return null;
        }
        transformedStyles.push(...transformed);
      }
      return transformedStyles;
    }

    if (isStyleShaderEntry(style) || isStyleRuleEntry(style)) {
      return null;
    }
    return this.transformFlatStyleForGpuStaticLabels_(
      /** @type {import('../../style/flat.js').FlatStyle} */ (style),
      features,
    );
  }

  /**
   * Convert one flat style with text into [baseStyleWithoutText, labelIconStyle].
   * @param {import('../../style/flat.js').FlatStyle} flatStyle Flat style.
   * @param {Array<import('../../Feature.js').FeatureLike>} features Features.
   * @return {Array<import('../../style/flat.js').FlatStyle>|null} One or two styles.
   * @private
   */
  transformFlatStyleForGpuStaticLabels_(flatStyle, features) {
    if (!hasTextStyleProperties(flatStyle)) {
      return [flatStyle];
    }

    const styleId = this.gpuStaticLabelStyleId_++;
    const useGlyphModeForStyle =
      this.gpuStaticLabelGlyphMode_ &&
      !('text-stroke-color' in flatStyle) &&
      !('text-stroke-width' in flatStyle) &&
      !('text-background-stroke-color' in flatStyle) &&
      !('text-background-stroke-width' in flatStyle);
    const atlasData = useGlyphModeForStyle
      ? this.buildGlyphLabelAtlasForStyle_(flatStyle, features, styleId)
      : this.buildStaticLabelAtlasForStyle_(flatStyle, features, styleId);
    if (!atlasData) {
      return null;
    }
    if (useGlyphModeForStyle) {
      this.gpuStaticLabelProperties_.push({
        properties: atlasData.properties,
      });
    } else {
      this.gpuStaticLabelProperties_.push({
        offsetXProperty: atlasData.offsetXProperty,
        offsetYProperty: atlasData.offsetYProperty,
        widthProperty: atlasData.widthProperty,
        heightProperty: atlasData.heightProperty,
        properties: [
          atlasData.offsetXProperty,
          atlasData.offsetYProperty,
          atlasData.widthProperty,
          atlasData.heightProperty,
        ],
      });
    }

    const transformed = [];
    const baseStyle = cloneStyleWithoutTextProperties(flatStyle);
    if (countOwnProperties(baseStyle) > 0) {
      transformed.push(baseStyle);
    }
    if (useGlyphModeForStyle) {
      transformed.push(
        ...this.createGlyphLabelIconStyles_(flatStyle, atlasData),
      );
    } else {
      transformed.push(this.createStaticLabelIconStyle_(flatStyle, atlasData));
    }
    return transformed;
  }

  /**
   * @typedef {Object} StaticLabelAtlasData
   * @property {HTMLCanvasElement|OffscreenCanvas} atlasImage Atlas image.
   * @property {number} atlasWidth Atlas width in pixels.
   * @property {number} atlasHeight Atlas height in pixels.
   * @property {string} offsetXProperty Feature property with atlas x.
   * @property {string} offsetYProperty Feature property with atlas y.
   * @property {string} widthProperty Feature property with atlas width.
   * @property {string} heightProperty Feature property with atlas height.
   */

  /**
   * Build one static text atlas for one flat style.
   * @param {import('../../style/flat.js').FlatStyle} flatStyle Flat style.
   * @param {Array<import('../../Feature.js').FeatureLike>} features Features.
   * @param {number} styleId Style id.
   * @return {StaticLabelAtlasData|null} Atlas data.
   * @private
   */
  buildStaticLabelAtlasForStyle_(flatStyle, features, styleId) {
    const textOnlyStyle = cloneTextOnlyStyle(flatStyle);

    const offsetXProperty = `olGpuStaticLabel${styleId}OffsetX`;
    const offsetYProperty = `olGpuStaticLabel${styleId}OffsetY`;
    const widthProperty = `olGpuStaticLabel${styleId}Width`;
    const heightProperty = `olGpuStaticLabel${styleId}Height`;
    const styleFunction = (() => {
      try {
        return flatStyleLikeToStyleFunction(textOnlyStyle, newParsingContext());
      } catch {
        return null;
      }
    })();
    if (!styleFunction) {
      return null;
    }

    const spriteByKey = new Map();
    const spriteByFeatureUid = new Map();
    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      if (typeof feature.set !== 'function') {
        return null;
      }
      const uid = getUid(feature);
      let styles;
      try {
        styles = styleFunction(feature, 1);
      } catch {
        return null;
      }
      if (!styles || !styles.length) {
        continue;
      }
      for (let j = 0; j < styles.length; j++) {
        const textStyle = styles[j].getText?.();
        if (!textStyle || textStyle.getPlacement?.() === 'line') {
          continue;
        }
        const textValue = normalizeTextValue(textStyle.getText?.());
        if (!textValue) {
          continue;
        }
        const sprite = this.getOrCreateStaticLabelSprite_(
          spriteByKey,
          textStyle,
          textValue,
        );
        if (!sprite) {
          continue;
        }
        spriteByFeatureUid.set(uid, sprite);
        break;
      }
    }

    const sprites = Array.from(spriteByKey.values());
    if (sprites.length && !this.packStaticLabelSprites_(sprites)) {
      return null;
    }

    const atlasImage = this.drawStaticLabelAtlas_(sprites);
    if (!atlasImage) {
      return null;
    }
    const atlasWidth = atlasImage.width;
    const atlasHeight = atlasImage.height;

    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      const sprite = spriteByFeatureUid.get(getUid(feature));
      if (sprite) {
        feature.set(offsetXProperty, sprite.x, true);
        feature.set(offsetYProperty, sprite.y, true);
        feature.set(widthProperty, sprite.width, true);
        feature.set(heightProperty, sprite.height, true);
      } else {
        feature.set(offsetXProperty, 0, true);
        feature.set(offsetYProperty, 0, true);
        feature.set(widthProperty, 0, true);
        feature.set(heightProperty, 0, true);
      }
    }

    return {
      atlasImage,
      atlasWidth,
      atlasHeight,
      offsetXProperty,
      offsetYProperty,
      widthProperty,
      heightProperty,
    };
  }

  /**
   * @typedef {Object} GlyphSlotProperties
   * @property {string} atlasOffsetXProperty Atlas x property.
   * @property {string} atlasOffsetYProperty Atlas y property.
   * @property {string} atlasWidthProperty Atlas width property.
   * @property {string} atlasHeightProperty Atlas height property.
   * @property {string} displacementXProperty Glyph displacement x property.
   * @property {string} displacementYProperty Glyph displacement y property.
   */

  /**
   * @typedef {Object} GlyphLabelAtlasData
   * @property {HTMLCanvasElement|OffscreenCanvas} atlasImage Atlas image.
   * @property {number} atlasWidth Atlas width in pixels.
   * @property {number} atlasHeight Atlas height in pixels.
   * @property {Array<GlyphSlotProperties>} slots Glyph slot descriptors.
   * @property {Array<string>} properties Feature properties used by glyph mode.
   * @property {string|null} backgroundWidthProperty Background width property.
   * @property {string|null} backgroundHeightProperty Background height property.
   * @property {string|null} backgroundDisplacementXProperty Background displacement x property.
   * @property {string|null} backgroundDisplacementYProperty Background displacement y property.
   */

  /**
   * Build one glyph atlas for one flat style. Labels are rendered as multiple icon passes.
   * This mode scales memory with unique glyphs instead of unique full label strings.
   * @param {import('../../style/flat.js').FlatStyle} flatStyle Flat style.
   * @param {Array<import('../../Feature.js').FeatureLike>} features Features.
   * @param {number} styleId Style id.
   * @return {GlyphLabelAtlasData|null} Glyph atlas data.
   * @private
   */
  buildGlyphLabelAtlasForStyle_(flatStyle, features, styleId) {
    const textOnlyStyle = cloneTextOnlyStyle(flatStyle);
    const styleFunction = (() => {
      try {
        return flatStyleLikeToStyleFunction(textOnlyStyle, newParsingContext());
      } catch {
        return null;
      }
    })();
    if (!styleFunction) {
      return null;
    }

    const glyphByKey = new Map();
    const featureDataByUid = new Map();
    let maxGlyphCount = 0;

    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      if (typeof feature.set !== 'function') {
        return null;
      }
      const uid = getUid(feature);
      let styles;
      try {
        styles = styleFunction(feature, 1);
      } catch {
        return null;
      }
      if (!styles || !styles.length) {
        continue;
      }

      for (let j = 0; j < styles.length; j++) {
        const textStyle = styles[j].getText?.();
        if (!textStyle || textStyle.getPlacement?.() === 'line') {
          continue;
        }
        const textValue = normalizeGlyphTextValue(textStyle.getText?.());
        if (!textValue) {
          continue;
        }

        const chars = Array.from(textValue);
        if (!chars.length) {
          continue;
        }

        const font = textStyle.getFont?.() || GPU_STATIC_LABEL_DEFAULT_FONT;
        const padding = normalizePadding(textStyle.getPadding?.());
        const glyphs = new Array(chars.length);
        let labelWidth = 0;
        let maxLineHeight = 1;

        for (let k = 0; k < chars.length; k++) {
          const glyph = this.getOrCreateGlyphSprite_(
            glyphByKey,
            font,
            chars[k],
          );
          if (!glyph) {
            continue;
          }
          glyphs[k] = glyph;
          labelWidth += glyph.advance;
          maxLineHeight = Math.max(maxLineHeight, glyph.lineHeight);
        }

        if (!glyphs.length) {
          continue;
        }

        maxGlyphCount = Math.max(maxGlyphCount, glyphs.length);
        const align = textStyle.getTextAlign?.();
        const baseline = textStyle.getTextBaseline?.();
        let textStartX = -labelWidth * 0.5;
        if (align === 'left' || align === 'start') {
          textStartX = 0;
        } else if (align === 'right' || align === 'end') {
          textStartX = -labelWidth;
        }

        let textTopY = -maxLineHeight * 0.5;
        if (baseline === 'top' || baseline === 'hanging') {
          textTopY = 0;
        } else if (
          baseline === 'bottom' ||
          baseline === 'alphabetic' ||
          baseline === 'ideographic'
        ) {
          textTopY = -maxLineHeight;
        }

        featureDataByUid.set(uid, {
          glyphs,
          textStartX,
          textTopY,
          labelWidth,
          labelHeight: maxLineHeight,
          padding,
        });
        break;
      }
    }

    if (!maxGlyphCount) {
      return null;
    }

    const glyphSprites = Array.from(glyphByKey.values());
    if (glyphSprites.length && !this.packStaticLabelSprites_(glyphSprites)) {
      return null;
    }

    const atlasImage = this.drawGlyphAtlas_(glyphSprites);
    if (!atlasImage) {
      return null;
    }

    const slotCount = Math.min(maxGlyphCount, GPU_GLYPH_MAX_LABEL_LENGTH);
    const slots = new Array(slotCount);
    const properties = [];
    for (let i = 0; i < slotCount; i++) {
      const prefix = `olGpuGlyph${styleId}Slot${i}`;
      const slot = {
        atlasOffsetXProperty: `${prefix}AtlasOffsetX`,
        atlasOffsetYProperty: `${prefix}AtlasOffsetY`,
        atlasWidthProperty: `${prefix}AtlasWidth`,
        atlasHeightProperty: `${prefix}AtlasHeight`,
        displacementXProperty: `${prefix}DisplacementX`,
        displacementYProperty: `${prefix}DisplacementY`,
      };
      slots[i] = slot;
      properties.push(
        slot.atlasOffsetXProperty,
        slot.atlasOffsetYProperty,
        slot.atlasWidthProperty,
        slot.atlasHeightProperty,
        slot.displacementXProperty,
        slot.displacementYProperty,
      );
    }

    let backgroundWidthProperty = null;
    let backgroundHeightProperty = null;
    let backgroundDisplacementXProperty = null;
    let backgroundDisplacementYProperty = null;
    if ('text-background-fill-color' in flatStyle) {
      backgroundWidthProperty = `olGpuGlyph${styleId}BackgroundWidth`;
      backgroundHeightProperty = `olGpuGlyph${styleId}BackgroundHeight`;
      backgroundDisplacementXProperty = `olGpuGlyph${styleId}BackgroundDisplacementX`;
      backgroundDisplacementYProperty = `olGpuGlyph${styleId}BackgroundDisplacementY`;
      properties.push(
        backgroundWidthProperty,
        backgroundHeightProperty,
        backgroundDisplacementXProperty,
        backgroundDisplacementYProperty,
      );
    }

    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      const uid = getUid(feature);
      const featureData = featureDataByUid.get(uid);

      if (!featureData) {
        for (let j = 0; j < slotCount; j++) {
          const slot = slots[j];
          feature.set(slot.atlasOffsetXProperty, 0, true);
          feature.set(slot.atlasOffsetYProperty, 0, true);
          feature.set(slot.atlasWidthProperty, 0, true);
          feature.set(slot.atlasHeightProperty, 0, true);
          feature.set(slot.displacementXProperty, 0, true);
          feature.set(slot.displacementYProperty, 0, true);
        }
        if (backgroundWidthProperty) {
          feature.set(backgroundWidthProperty, 0, true);
          feature.set(backgroundHeightProperty, 0, true);
          feature.set(backgroundDisplacementXProperty, 0, true);
          feature.set(backgroundDisplacementYProperty, 0, true);
        }
        continue;
      }

      let cursorX = 0;
      for (let j = 0; j < slotCount; j++) {
        const slot = slots[j];
        const glyph = featureData.glyphs[j];
        if (!glyph) {
          feature.set(slot.atlasOffsetXProperty, 0, true);
          feature.set(slot.atlasOffsetYProperty, 0, true);
          feature.set(slot.atlasWidthProperty, 0, true);
          feature.set(slot.atlasHeightProperty, 0, true);
          feature.set(slot.displacementXProperty, 0, true);
          feature.set(slot.displacementYProperty, 0, true);
          continue;
        }

        feature.set(slot.atlasOffsetXProperty, glyph.x, true);
        feature.set(slot.atlasOffsetYProperty, glyph.y, true);
        feature.set(slot.atlasWidthProperty, glyph.width, true);
        feature.set(slot.atlasHeightProperty, glyph.height, true);
        feature.set(
          slot.displacementXProperty,
          featureData.textStartX + cursorX - glyph.sdfPadding,
          true,
        );
        feature.set(
          slot.displacementYProperty,
          featureData.textTopY - glyph.sdfPadding,
          true,
        );
        cursorX += glyph.advance;
      }

      if (backgroundWidthProperty) {
        const backgroundWidth =
          featureData.labelWidth +
          featureData.padding[1] +
          featureData.padding[3];
        const backgroundHeight =
          featureData.labelHeight +
          featureData.padding[0] +
          featureData.padding[2];
        feature.set(backgroundWidthProperty, backgroundWidth, true);
        feature.set(backgroundHeightProperty, backgroundHeight, true);
        feature.set(
          backgroundDisplacementXProperty,
          featureData.textStartX - featureData.padding[3],
          true,
        );
        feature.set(
          backgroundDisplacementYProperty,
          featureData.textTopY - featureData.padding[0],
          true,
        );
      }
    }

    return {
      atlasImage,
      atlasWidth: atlasImage.width,
      atlasHeight: atlasImage.height,
      slots,
      properties,
      backgroundWidthProperty,
      backgroundHeightProperty,
      backgroundDisplacementXProperty,
      backgroundDisplacementYProperty,
    };
  }

  /**
   * @param {Map<string, Object>} glyphByKey Glyph cache.
   * @param {string} font Font.
   * @param {string} character One character.
   * @return {Object|null} Glyph sprite descriptor.
   * @private
   */
  getOrCreateGlyphSprite_(glyphByKey, font, character) {
    const key = `${font}::${character}`;
    if (glyphByKey.has(key)) {
      return glyphByKey.get(key);
    }

    const context = this.gpuStaticLabelContext_;
    context.font = font;
    const metrics = context.measureText(character);
    const fontSize = parseFontSizePx(font);
    const ascent = Math.max(
      1,
      Math.ceil(metrics.actualBoundingBoxAscent || fontSize * 0.8),
    );
    const descent = Math.max(
      0,
      Math.ceil(metrics.actualBoundingBoxDescent || fontSize * 0.2),
    );
    const glyphWidth = Math.max(
      1,
      Math.ceil(
        metrics.actualBoundingBoxLeft !== undefined &&
          metrics.actualBoundingBoxRight !== undefined
          ? metrics.actualBoundingBoxLeft + metrics.actualBoundingBoxRight
          : metrics.width,
      ),
    );
    const advance = Math.max(1, Math.ceil(metrics.width));
    const lineHeight = Math.max(
      1,
      Math.ceil(fontSize * GPU_STATIC_LABEL_LINE_HEIGHT_FACTOR),
    );
    const sdfPadding = GPU_GLYPH_BITMAP_PADDING + GPU_GLYPH_SDF_RADIUS;
    const glyph = {
      key,
      character,
      font,
      advance,
      lineHeight,
      width: glyphWidth + 2 * sdfPadding,
      height: Math.max(lineHeight, ascent + descent) + 2 * sdfPadding,
      sdfPadding,
      baselineY: sdfPadding + ascent,
      x: 0,
      y: 0,
    };
    glyphByKey.set(key, glyph);
    return glyph;
  }

  /**
   * Rasterize one glyph to a monochannel SDF image.
   * @param {Object} glyph Glyph descriptor.
   * @return {ImageData|null} SDF image.
   * @private
   */
  rasterizeGlyphSdfImageData_(glyph) {
    const context = this.gpuStaticLabelGlyphRasterContext_;
    const width = glyph.width;
    const height = glyph.height;
    if (width <= 0 || height <= 0) {
      return null;
    }

    context.canvas.width = width;
    context.canvas.height = height;
    context.clearRect(0, 0, width, height);
    context.font = glyph.font;
    context.textAlign = 'left';
    context.textBaseline = 'alphabetic';
    context.fillStyle = '#ffffff';
    context.fillText(glyph.character, glyph.sdfPadding, glyph.baselineY);

    const image = context.getImageData(0, 0, width, height);
    const pixelCount = width * height;
    const binaryMask = new Uint8Array(pixelCount);
    const data = image.data;
    for (let i = 0; i < pixelCount; i++) {
      binaryMask[i] = data[i * 4 + 3] > 127 ? 1 : 0;
    }

    const squaredToOutside = computeSquaredDistanceTransform(
      binaryMask,
      width,
      height,
      0,
    );
    const squaredToInside = computeSquaredDistanceTransform(
      binaryMask,
      width,
      height,
      1,
    );
    const radius = GPU_GLYPH_SDF_RADIUS;
    const normalizer = 1 / (2 * radius);
    const sdfData = new Uint8ClampedArray(pixelCount * 4);
    for (let i = 0; i < pixelCount; i++) {
      const signedDistance =
        Math.sqrt(squaredToOutside[i]) - Math.sqrt(squaredToInside[i]);
      const sdf = Math.max(
        0,
        Math.min(1, GPU_GLYPH_SDF_CUTOFF + signedDistance * normalizer),
      );
      const alpha = Math.round(sdf * 255);
      const offset = i * 4;
      sdfData[offset] = 255;
      sdfData[offset + 1] = 255;
      sdfData[offset + 2] = 255;
      sdfData[offset + 3] = alpha;
    }
    return new ImageData(sdfData, width, height);
  }

  /**
   * Draw all glyph sprites into a packed atlas.
   * @param {Array<Object>} glyphs Packed glyph descriptors.
   * @return {HTMLCanvasElement|OffscreenCanvas|null} Atlas image.
   * @private
   */
  drawGlyphAtlas_(glyphs) {
    if (!glyphs.length) {
      this.gpuStaticLabelContext_.canvas.width = 1;
      this.gpuStaticLabelContext_.canvas.height = 1;
      this.gpuStaticLabelContext_.clearRect(0, 0, 1, 1);
    }

    const context = this.gpuStaticLabelContext_;
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    for (let i = 0; i < glyphs.length; i++) {
      const glyph = glyphs[i];
      const imageData = this.rasterizeGlyphSdfImageData_(glyph);
      if (!imageData) {
        continue;
      }
      context.putImageData(imageData, glyph.x, glyph.y);
    }

    const sourceCanvas = context.canvas;
    if (sourceCanvas.width <= 0 || sourceCanvas.height <= 0) {
      return null;
    }
    const atlasContext = createCanvasContext2D(
      sourceCanvas.width,
      sourceCanvas.height,
    );
    atlasContext.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
    atlasContext.drawImage(sourceCanvas, 0, 0);
    const atlasImage = atlasContext.canvas;
    this.gpuStaticLabelAtlases_.push(atlasImage);
    return atlasImage;
  }

  /**
   * @param {Map<string, Object>} spriteByKey Sprite cache.
   * @param {import('../../style/Text.js').default} textStyle Text style.
   * @param {string} textValue Text.
   * @return {Object|null} Sprite descriptor.
   * @private
   */
  getOrCreateStaticLabelSprite_(spriteByKey, textStyle, textValue) {
    const fill = textStyle.getFill?.();
    const stroke = textStyle.getStroke?.();
    const backgroundFill = textStyle.getBackgroundFill?.();
    const backgroundStroke = textStyle.getBackgroundStroke?.();
    const fillStyle = toCanvasStyleColor(fill?.getColor?.());
    const strokeStyle = toCanvasStyleColor(stroke?.getColor?.());
    const backgroundFillStyle = toCanvasStyleColor(
      backgroundFill?.getColor?.(),
    );
    const backgroundStrokeStyle = toCanvasStyleColor(
      backgroundStroke?.getColor?.(),
    );

    const strokeWidth = stroke?.getWidth?.() || 0;
    const backgroundStrokeWidth = backgroundStroke?.getWidth?.() || 0;
    if (
      !fillStyle &&
      (!strokeStyle || strokeWidth <= 0) &&
      !backgroundFillStyle &&
      (!backgroundStrokeStyle || backgroundStrokeWidth <= 0)
    ) {
      return null;
    }

    const font = textStyle.getFont?.() || GPU_STATIC_LABEL_DEFAULT_FONT;
    const padding = normalizePadding(textStyle.getPadding?.());
    const spriteKey = JSON.stringify({
      t: textValue,
      f: font,
      fill: String(fillStyle || ''),
      stroke: String(strokeStyle || ''),
      sw: strokeWidth,
      slc: stroke?.getLineCap?.() || '',
      slj: stroke?.getLineJoin?.() || '',
      sm: stroke?.getMiterLimit?.() || '',
      sld: stroke?.getLineDash?.() || null,
      sldo: stroke?.getLineDashOffset?.() || 0,
      bgf: String(backgroundFillStyle || ''),
      bgs: String(backgroundStrokeStyle || ''),
      bgsw: backgroundStrokeWidth,
      bgslc: backgroundStroke?.getLineCap?.() || '',
      bgslj: backgroundStroke?.getLineJoin?.() || '',
      bgsm: backgroundStroke?.getMiterLimit?.() || '',
      bgsld: backgroundStroke?.getLineDash?.() || null,
      bgsldo: backgroundStroke?.getLineDashOffset?.() || 0,
      p: padding,
    });
    if (spriteByKey.has(spriteKey)) {
      return spriteByKey.get(spriteKey);
    }

    const context = this.gpuStaticLabelContext_;
    context.font = font;
    const lines = textValue.split('\n');
    let maxWidth = 0;
    for (let i = 0; i < lines.length; i++) {
      const width = context.measureText(lines[i]).width;
      if (width > maxWidth) {
        maxWidth = width;
      }
    }

    const lineHeight = Math.max(
      1,
      Math.ceil(parseFontSizePx(font) * GPU_STATIC_LABEL_LINE_HEIGHT_FACTOR),
    );
    const textWidth = Math.max(1, Math.ceil(maxWidth));
    const textHeight = Math.max(1, Math.ceil(lineHeight * lines.length));
    const textStrokePadding = strokeWidth > 0 ? Math.ceil(strokeWidth) : 0;
    const backgroundStrokePadding =
      backgroundStrokeWidth > 0 ? Math.ceil(backgroundStrokeWidth) : 0;
    const textOffsetX =
      backgroundStrokePadding + textStrokePadding + padding[3];
    const textOffsetY =
      backgroundStrokePadding + textStrokePadding + padding[0];
    const width =
      textWidth +
      padding[1] +
      padding[3] +
      2 * textStrokePadding +
      2 * backgroundStrokePadding;
    const height =
      textHeight +
      padding[0] +
      padding[2] +
      2 * textStrokePadding +
      2 * backgroundStrokePadding;

    const sprite = {
      key: spriteKey,
      lines,
      font,
      width: Math.max(1, Math.ceil(width)),
      height: Math.max(1, Math.ceil(height)),
      lineHeight,
      textOffsetX,
      textOffsetY,
      fillStyle,
      strokeStyle,
      strokeWidth,
      strokeLineCap: stroke?.getLineCap?.(),
      strokeLineJoin: stroke?.getLineJoin?.(),
      strokeMiterLimit: stroke?.getMiterLimit?.(),
      strokeLineDash: stroke?.getLineDash?.(),
      strokeLineDashOffset: stroke?.getLineDashOffset?.(),
      backgroundFillStyle,
      backgroundStrokeStyle,
      backgroundStrokeWidth,
      backgroundStrokeLineCap: backgroundStroke?.getLineCap?.(),
      backgroundStrokeLineJoin: backgroundStroke?.getLineJoin?.(),
      backgroundStrokeMiterLimit: backgroundStroke?.getMiterLimit?.(),
      backgroundStrokeLineDash: backgroundStroke?.getLineDash?.(),
      backgroundStrokeLineDashOffset: backgroundStroke?.getLineDashOffset?.(),
      x: 0,
      y: 0,
    };
    spriteByKey.set(spriteKey, sprite);
    return sprite;
  }

  /**
   * Place sprites into one atlas.
   * @param {Array<Object>} sprites Sprite descriptors.
   * @return {boolean} Whether packing succeeded.
   * @private
   */
  packStaticLabelSprites_(sprites) {
    const maxAtlasSize = this.gpuStaticLabelMaxAtlasSize_;
    const sorted = sprites.slice().sort((a, b) => b.height - a.height);
    const totalArea = sorted.reduce((sum, sprite) => {
      return (
        sum +
        (sprite.width + 2 * GPU_STATIC_LABEL_SHELF_PADDING) *
          (sprite.height + 2 * GPU_STATIC_LABEL_SHELF_PADDING)
      );
    }, 0);
    const maxSpriteWidth = sorted.reduce((max, sprite) => {
      return Math.max(max, sprite.width + 2 * GPU_STATIC_LABEL_SHELF_PADDING);
    }, 0);
    let atlasWidth = Math.max(
      GPU_STATIC_LABEL_MIN_ATLAS_SIZE,
      nextPowerOfTwo(Math.ceil(Math.sqrt(totalArea))),
      nextPowerOfTwo(maxSpriteWidth + 2 * GPU_STATIC_LABEL_ATLAS_PADDING),
    );
    if (atlasWidth > maxAtlasSize) {
      return false;
    }

    while (atlasWidth <= maxAtlasSize) {
      let x = GPU_STATIC_LABEL_ATLAS_PADDING;
      let y = GPU_STATIC_LABEL_ATLAS_PADDING;
      let rowHeight = 0;
      let requiredHeight = 0;
      let failed = false;

      for (let i = 0; i < sorted.length; i++) {
        const sprite = sorted[i];
        const packedWidth = sprite.width + 2 * GPU_STATIC_LABEL_SHELF_PADDING;
        const packedHeight = sprite.height + 2 * GPU_STATIC_LABEL_SHELF_PADDING;

        if (x + packedWidth + GPU_STATIC_LABEL_ATLAS_PADDING > atlasWidth) {
          x = GPU_STATIC_LABEL_ATLAS_PADDING;
          y += rowHeight;
          rowHeight = 0;
        }

        if (y + packedHeight + GPU_STATIC_LABEL_ATLAS_PADDING > maxAtlasSize) {
          failed = true;
          break;
        }

        sprite.x = x + GPU_STATIC_LABEL_SHELF_PADDING;
        sprite.y = y + GPU_STATIC_LABEL_SHELF_PADDING;
        x += packedWidth;
        rowHeight = Math.max(rowHeight, packedHeight);
        requiredHeight = Math.max(
          requiredHeight,
          y + rowHeight + GPU_STATIC_LABEL_ATLAS_PADDING,
        );
      }

      if (!failed) {
        const atlasHeight = Math.max(
          GPU_STATIC_LABEL_MIN_ATLAS_SIZE,
          nextPowerOfTwo(requiredHeight),
        );
        if (atlasHeight <= maxAtlasSize) {
          this.gpuStaticLabelContext_.canvas.width = atlasWidth;
          this.gpuStaticLabelContext_.canvas.height = atlasHeight;
          return true;
        }
      }

      atlasWidth *= 2;
    }
    return false;
  }

  /**
   * Draw all sprites into the packed atlas.
   * @param {Array<Object>} sprites Packed sprite descriptors.
   * @return {HTMLCanvasElement|OffscreenCanvas|null} Atlas image.
   * @private
   */
  drawStaticLabelAtlas_(sprites) {
    if (!sprites.length) {
      this.gpuStaticLabelContext_.canvas.width = 1;
      this.gpuStaticLabelContext_.canvas.height = 1;
      this.gpuStaticLabelContext_.clearRect(0, 0, 1, 1);
    }

    const context = this.gpuStaticLabelContext_;
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    context.textAlign = 'left';
    context.textBaseline = 'top';

    for (let i = 0; i < sprites.length; i++) {
      const sprite = sprites[i];
      const x = sprite.x;
      const y = sprite.y;

      if (sprite.backgroundFillStyle) {
        context.fillStyle = sprite.backgroundFillStyle;
        context.fillRect(x, y, sprite.width, sprite.height);
      }
      if (sprite.backgroundStrokeStyle && sprite.backgroundStrokeWidth > 0) {
        context.strokeStyle = sprite.backgroundStrokeStyle;
        context.lineWidth = sprite.backgroundStrokeWidth;
        context.lineCap = sprite.backgroundStrokeLineCap || 'round';
        context.lineJoin = sprite.backgroundStrokeLineJoin || 'round';
        context.miterLimit = sprite.backgroundStrokeMiterLimit || 10;
        context.setLineDash(sprite.backgroundStrokeLineDash || []);
        context.lineDashOffset = sprite.backgroundStrokeLineDashOffset || 0;
        const inset = sprite.backgroundStrokeWidth * 0.5;
        context.strokeRect(
          x + inset,
          y + inset,
          Math.max(0, sprite.width - sprite.backgroundStrokeWidth),
          Math.max(0, sprite.height - sprite.backgroundStrokeWidth),
        );
      }

      context.font = sprite.font;
      if (sprite.strokeStyle && sprite.strokeWidth > 0) {
        context.strokeStyle = sprite.strokeStyle;
        context.lineWidth = sprite.strokeWidth;
        context.lineCap = sprite.strokeLineCap || 'round';
        context.lineJoin = sprite.strokeLineJoin || 'round';
        context.miterLimit = sprite.strokeMiterLimit || 10;
        context.setLineDash(sprite.strokeLineDash || []);
        context.lineDashOffset = sprite.strokeLineDashOffset || 0;
      } else {
        context.setLineDash([]);
      }
      if (sprite.fillStyle) {
        context.fillStyle = sprite.fillStyle;
      }

      for (let j = 0; j < sprite.lines.length; j++) {
        const line = sprite.lines[j];
        const drawX = x + sprite.textOffsetX;
        const drawY = y + sprite.textOffsetY + j * sprite.lineHeight;
        if (sprite.strokeStyle && sprite.strokeWidth > 0) {
          context.strokeText(line, drawX, drawY);
        }
        if (sprite.fillStyle) {
          context.fillText(line, drawX, drawY);
        }
      }
    }

    const sourceCanvas = context.canvas;
    if (sourceCanvas.width <= 0 || sourceCanvas.height <= 0) {
      return null;
    }
    const atlasContext = createCanvasContext2D(
      sourceCanvas.width,
      sourceCanvas.height,
    );
    atlasContext.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
    atlasContext.drawImage(sourceCanvas, 0, 0);
    const atlasImage = atlasContext.canvas;
    this.gpuStaticLabelAtlases_.push(atlasImage);
    return atlasImage;
  }

  /**
   * Create icon style that uses static atlas values stored in feature properties.
   * @param {import('../../style/flat.js').FlatStyle} flatStyle Source flat style with text keys.
   * @param {StaticLabelAtlasData} atlasData Atlas data.
   * @return {import('../../style/flat.js').FlatStyle} Icon style.
   * @private
   */
  createStaticLabelIconStyle_(flatStyle, atlasData) {
    const align =
      typeof flatStyle['text-align'] === 'string'
        ? flatStyle['text-align']
        : undefined;
    const baseline =
      typeof flatStyle['text-baseline'] === 'string'
        ? flatStyle['text-baseline']
        : undefined;
    const labelStyle = {
      'icon-image': atlasData.atlasImage,
      'icon-offset': [
        'array',
        ['get', atlasData.offsetXProperty],
        ['get', atlasData.offsetYProperty],
      ],
      'icon-size': [
        'array',
        ['get', atlasData.widthProperty],
        ['get', atlasData.heightProperty],
      ],
      'icon-width': atlasData.atlasWidth,
      'icon-height': atlasData.atlasHeight,
      'icon-anchor': [
        textAlignToAnchorX(align),
        textBaselineToAnchorY(baseline),
      ],
      'icon-anchor-x-units': 'fraction',
      'icon-anchor-y-units': 'fraction',
    };

    const displacement = createTextDisplacementExpression(
      flatStyle['text-offset-x'],
      flatStyle['text-offset-y'],
    );
    if (displacement) {
      labelStyle['icon-displacement'] = displacement;
    }
    if ('text-scale' in flatStyle) {
      labelStyle['icon-scale'] = flatStyle['text-scale'];
    }
    if ('text-rotation' in flatStyle) {
      labelStyle['icon-rotation'] = flatStyle['text-rotation'];
    }
    if ('text-rotate-with-view' in flatStyle) {
      labelStyle['icon-rotate-with-view'] = flatStyle['text-rotate-with-view'];
    }
    if ('text-declutter-mode' in flatStyle) {
      labelStyle['icon-declutter-mode'] = flatStyle['text-declutter-mode'];
    }
    if ('z-index' in flatStyle) {
      labelStyle['z-index'] = flatStyle['z-index'];
    }
    return labelStyle;
  }

  /**
   * @return {HTMLCanvasElement|OffscreenCanvas} 1x1 white pixel canvas.
   * @private
   */
  getGpuStaticLabelWhitePixelAtlas_() {
    if (this.gpuStaticLabelWhitePixelAtlas_) {
      return this.gpuStaticLabelWhitePixelAtlas_;
    }
    const context = createCanvasContext2D(1, 1);
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, 1, 1);
    this.gpuStaticLabelWhitePixelAtlas_ = context.canvas;
    this.gpuStaticLabelAtlases_.push(context.canvas);
    return this.gpuStaticLabelWhitePixelAtlas_;
  }

  /**
   * @param {string} displacementXProperty X displacement property.
   * @param {string} displacementYProperty Y displacement property.
   * @param {import('../../expr/expression.js').ExpressionValue|undefined} textOffsetX Text offset x.
   * @param {import('../../expr/expression.js').ExpressionValue|undefined} textOffsetY Text offset y.
   * @return {Array<*>} Displacement expression.
   * @private
   */
  createGlyphDisplacementExpression_(
    displacementXProperty,
    displacementYProperty,
    textOffsetX,
    textOffsetY,
  ) {
    const xExpression =
      textOffsetX === undefined
        ? ['get', displacementXProperty]
        : ['+', ['get', displacementXProperty], textOffsetX];
    const yExpression =
      textOffsetY === undefined
        ? ['get', displacementYProperty]
        : ['+', ['get', displacementYProperty], textOffsetY];
    return ['array', xExpression, negateNumberExpression(yExpression)];
  }

  /**
   * Create icon styles for glyph-atlas mode.
   * @param {import('../../style/flat.js').FlatStyle} flatStyle Source flat style with text keys.
   * @param {GlyphLabelAtlasData} atlasData Glyph atlas data.
   * @return {Array<import('../../style/flat.js').FlatStyle>} Icon styles.
   * @private
   */
  createGlyphLabelIconStyles_(flatStyle, atlasData) {
    const styles = [];
    const textOffsetX = flatStyle['text-offset-x'];
    const textOffsetY = flatStyle['text-offset-y'];
    if (
      atlasData.backgroundWidthProperty &&
      atlasData.backgroundHeightProperty &&
      atlasData.backgroundDisplacementXProperty &&
      atlasData.backgroundDisplacementYProperty &&
      'text-background-fill-color' in flatStyle
    ) {
      const backgroundStyle = {
        'icon-image': this.getGpuStaticLabelWhitePixelAtlas_(),
        'icon-offset': [0, 0],
        'icon-size': [
          'array',
          ['get', atlasData.backgroundWidthProperty],
          ['get', atlasData.backgroundHeightProperty],
        ],
        'icon-width': 1,
        'icon-height': 1,
        'icon-anchor': [0, 0],
        'icon-anchor-x-units': 'fraction',
        'icon-anchor-y-units': 'fraction',
        'icon-displacement': this.createGlyphDisplacementExpression_(
          atlasData.backgroundDisplacementXProperty,
          atlasData.backgroundDisplacementYProperty,
          textOffsetX,
          textOffsetY,
        ),
        'icon-color': flatStyle['text-background-fill-color'],
      };
      if ('z-index' in flatStyle) {
        backgroundStyle['z-index'] = flatStyle['z-index'];
      }
      styles.push(backgroundStyle);
    }

    for (let i = 0; i < atlasData.slots.length; i++) {
      const slot = atlasData.slots[i];
      const labelStyle = {
        'icon-image': atlasData.atlasImage,
        'icon-offset': [
          'array',
          ['get', slot.atlasOffsetXProperty],
          ['get', slot.atlasOffsetYProperty],
        ],
        'icon-size': [
          'array',
          ['get', slot.atlasWidthProperty],
          ['get', slot.atlasHeightProperty],
        ],
        'icon-width': atlasData.atlasWidth,
        'icon-height': atlasData.atlasHeight,
        'icon-anchor': [0, 0],
        'icon-anchor-x-units': 'fraction',
        'icon-anchor-y-units': 'fraction',
        'icon-displacement': this.createGlyphDisplacementExpression_(
          slot.displacementXProperty,
          slot.displacementYProperty,
          textOffsetX,
          textOffsetY,
        ),
        'icon-color':
          'text-fill-color' in flatStyle
            ? flatStyle['text-fill-color']
            : GPU_GLYPH_DEFAULT_FILL_COLOR,
        'icon-sdf': true,
        'icon-sdf-cutoff': GPU_GLYPH_SDF_CUTOFF,
        'icon-sdf-smoothing': GPU_GLYPH_SDF_SMOOTHING_PX,
      };
      if ('text-scale' in flatStyle) {
        labelStyle['icon-scale'] = flatStyle['text-scale'];
      }
      if ('text-rotation' in flatStyle) {
        labelStyle['icon-rotation'] = flatStyle['text-rotation'];
      }
      if ('text-rotate-with-view' in flatStyle) {
        labelStyle['icon-rotate-with-view'] =
          flatStyle['text-rotate-with-view'];
      }
      if ('text-declutter-mode' in flatStyle) {
        labelStyle['icon-declutter-mode'] = flatStyle['text-declutter-mode'];
      }
      if ('z-index' in flatStyle) {
        labelStyle['z-index'] = flatStyle['z-index'];
      }
      styles.push(labelStyle);
    }
    return styles;
  }

  /**
   * @param {Options} options Options.
   * @private
   */
  applyOptions_(options) {
    this.styleVariables_ = options.variables;
    this.sourceStyle_ = options.style;
    this.style_ = options.style;
    const staticLabelOption = options.gpuStaticLabel;
    if (staticLabelOption === 'soft' || staticLabelOption === 'glyph-soft') {
      this.gpuStaticLabelMode_ = 'soft';
    } else if (
      staticLabelOption === true ||
      staticLabelOption === 'strict' ||
      staticLabelOption === 'glyph-strict'
    ) {
      this.gpuStaticLabelMode_ = 'strict';
    } else {
      this.gpuStaticLabelMode_ = 'off';
    }
    this.gpuStaticLabelGlyphMode_ =
      staticLabelOption === 'glyph-soft' ||
      staticLabelOption === 'glyph-strict';
    this.gpuStaticLabelEnabled_ = this.gpuStaticLabelMode_ !== 'off';
    this.gpuStaticLabelSoftMode_ = this.gpuStaticLabelMode_ === 'soft';
    this.gpuStaticLabelReady_ = false;
    this.gpuStaticLabelInitAttempted_ = false;
    this.gpuStaticLabelStyleId_ = 0;
    this.gpuStaticLabelRefreshPending_ = false;
    this.gpuStaticLabelProperties_.length = 0;
    this.releaseGpuStaticLabelAtlases_();
    const throttle = options.textRenderThrottleMs;
    if (throttle === undefined || throttle === null || throttle === 'auto') {
      this.textRenderThrottleAuto_ = true;
      this.textRenderThrottleMs_ = DEFAULT_TEXT_REBUILD_THROTTLE_MS;
      this.textOverlayRenderThrottleMs_ =
        DEFAULT_TEXT_OVERLAY_RENDER_THROTTLE_MS;
    } else {
      this.textRenderThrottleAuto_ = false;
      this.textRenderThrottleMs_ = Math.max(0, throttle);
      this.textOverlayRenderThrottleMs_ = Math.max(0, throttle);
    }
    this.textRebuildNeeded_ = false;
    this.textRebuildQueued_ = false;
    this.textRebuildInFlight_ = false;
    this.lastTextRenderTime_ = -Infinity;
    this.lastTextOverlayRenderTime_ = -Infinity;
    this.textRebuildDurationAvgMs_ = 0;
    this.textRenderDurationAvgMs_ = 0;
    this.currentFpsAvg_ = 60;
    this.lastRenderedTextInstructionsKey_ = null;
    if (this.textRebuildTimerId_) {
      clearTimeout(this.textRebuildTimerId_);
      this.textRebuildTimerId_ = 0;
    }
  }

  /**
   * @private
   */
  createRenderers_() {
    if (this.buffers_) {
      this.disposeBuffers(this.buffers_);
      this.buffers_ = null;
    }
    if (this.styleRenderer_) {
      this.flushPendingTextInstructions_();
      this.styleRenderer_.dispose();
    }
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
    const gl = this.helper.getGL();
    const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    if (Number.isFinite(maxTextureSize) && maxTextureSize > 0) {
      this.gpuStaticLabelMaxAtlasSize_ = Math.min(
        GPU_STATIC_LABEL_MAX_ATLAS_SIZE,
        Math.floor(maxTextureSize),
      );
    } else {
      this.gpuStaticLabelMaxAtlasSize_ = GPU_STATIC_LABEL_MAX_ATLAS_SIZE;
    }

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
    if (this.gpuStaticLabelReady_) {
      this.initializeStaticLabelFeatureProperties_(feature);
      this.queueGpuStaticLabelRefresh_();
    }
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
    } else {
      // In soft mode, property-only updates (e.g. text content) should refresh atlas.
      this.queueGpuStaticLabelRefresh_();
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
        this.lastPointAnimationUpdateTime_ = nowMs();
        // MixedGeometryBatch stores a reference to the Point's flatCoordinates array,
        // so we don't need to call `changeFeature()` for coordinate-only animations.
        this.dirtyPointUids_.add(uid);
        if (
          this.buffers_?.textInstructionsKey &&
          this.textRenderThrottleMs_ > 0
        ) {
          if (
            !this.textRebuildNeeded_ &&
            this.isPointPotentiallyVisibleForText_(
              pointGeometry.getFlatCoordinates(),
            )
          ) {
            this.textRebuildNeeded_ = true;
          }
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
    this.queueGpuStaticLabelRefresh_();
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
    this.queueGpuStaticLabelRefresh_();
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
   * Check whether a point can contribute to text rendering for the current view.
   * Coordinates are tested in clip space against the viewport expanded by a small
   * padding to avoid pop-in at screen edges.
   * @param {Array<number>} flatCoordinates Point flat coordinates.
   * @return {boolean} Whether the point is potentially visible for text rendering.
   * @private
   */
  isPointPotentiallyVisibleForText_(flatCoordinates) {
    if (!flatCoordinates || flatCoordinates.length < 2) {
      return false;
    }
    const t = this.renderTransform_;
    const clipX = t[0] * flatCoordinates[0] + t[2] * flatCoordinates[1] + t[4];
    const clipY = t[1] * flatCoordinates[0] + t[3] * flatCoordinates[1] + t[5];
    if (!Number.isFinite(clipX) || !Number.isFinite(clipY)) {
      return true;
    }
    const min = -1 - TEXT_REBUILD_VISIBILITY_PADDING_CLIP;
    const max = 1 + TEXT_REBUILD_VISIBILITY_PADDING_CLIP;
    return clipX >= min && clipX <= max && clipY >= min && clipY <= max;
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
          const viewMoving =
            frameState.viewHints[ViewHint.ANIMATING] ||
            frameState.viewHints[ViewHint.INTERACTING];
          this.updateTextRenderThrottle_(currentFps, viewMoving);
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
   * Adaptive throttling for text rebuild and overlay render.
   * Rebuilds can be slower than overlay blits to avoid visible flicker.
   * @param {number} currentFps Current map FPS.
   * @param {boolean} viewMoving Whether view is animating/interacting.
   * @private
   */
  updateTextRenderThrottle_(currentFps, viewMoving) {
    this.currentFpsAvg_ = this.currentFpsAvg_
      ? this.currentFpsAvg_ * (1 - TEXT_RENDER_FPS_SMOOTHING) +
        currentFps * TEXT_RENDER_FPS_SMOOTHING
      : currentFps;
    const fps = this.currentFpsAvg_ || currentFps;
    const performanceFactor =
      fps < CRITICAL_FPS_THRESHOLD
        ? Math.max(0.2, fps / CRITICAL_FPS_THRESHOLD)
        : 1;
    const rebuildCostMs = this.textRebuildDurationAvgMs_ * 1.1;
    const overlayRenderCostMs = this.textRenderDurationAvgMs_ * 1.05;

    if (rebuildCostMs > 0) {
      const rebuildDutyCycle = TARGET_REBUILD_DUTY_CYCLE * performanceFactor;
      const rebuildInterval = rebuildCostMs / rebuildDutyCycle - rebuildCostMs;
      const clampedRebuildInterval = Math.max(
        MIN_TEXT_RENDER_THROTTLE_MS,
        Math.min(MAX_TEXT_REBUILD_THROTTLE_MS, rebuildInterval),
      );
      this.textRenderThrottleMs_ =
        this.textRenderThrottleMs_ * (1 - TEXT_RENDER_THROTTLE_SMOOTHING) +
        clampedRebuildInterval * TEXT_RENDER_THROTTLE_SMOOTHING;
    } else {
      this.textRenderThrottleMs_ = DEFAULT_TEXT_REBUILD_THROTTLE_MS;
    }

    if (overlayRenderCostMs > 0) {
      const overlayDutyCycle =
        TARGET_OVERLAY_RENDER_DUTY_CYCLE * performanceFactor;
      const overlayInterval =
        overlayRenderCostMs / overlayDutyCycle - overlayRenderCostMs;
      const maxOverlayThrottle = viewMoving
        ? MAX_TEXT_OVERLAY_RENDER_THROTTLE_MOVING_MS
        : MAX_TEXT_OVERLAY_RENDER_THROTTLE_MS;
      const clampedOverlayInterval = Math.max(
        MIN_TEXT_RENDER_THROTTLE_MS,
        Math.min(maxOverlayThrottle, overlayInterval),
      );
      this.textOverlayRenderThrottleMs_ =
        this.textOverlayRenderThrottleMs_ *
          (1 - TEXT_RENDER_THROTTLE_SMOOTHING) +
        clampedOverlayInterval * TEXT_RENDER_THROTTLE_SMOOTHING;
    } else {
      this.textOverlayRenderThrottleMs_ = viewMoving
        ? Math.min(
            DEFAULT_TEXT_OVERLAY_RENDER_THROTTLE_MS,
            MAX_TEXT_OVERLAY_RENDER_THROTTLE_MOVING_MS,
          )
        : DEFAULT_TEXT_OVERLAY_RENDER_THROTTLE_MS;
    }
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
      this.lastRenderedTextInstructionsKey_ = null;
      this.flushPendingTextInstructions_();
      return;
    }

    if (!this.buffers_?.textInstructionsKey) {
      // If buffers are temporarily unavailable, keep the previous overlay
      // to avoid visible flicker during rebuilds.
      return;
    }

    const currentTextInstructionsKey = this.buffers_.textInstructionsKey;
    const keyChanged =
      currentTextInstructionsKey !== this.lastRenderedTextInstructionsKey_;
    const now = frameState.time;
    if (
      this.textOverlayRenderThrottleMs_ > 0 &&
      !keyChanged &&
      now - this.lastTextOverlayRenderTime_ < this.textOverlayRenderThrottleMs_
    ) {
      return;
    }
    this.lastTextOverlayRenderTime_ = now;

    const renderStart = nowMs();
    this.styleRenderer_.finalizeTextRender(frameState).then(() => {
      if (this.buffers_?.textInstructionsKey === currentTextInstructionsKey) {
        this.lastRenderedTextInstructionsKey_ = currentTextInstructionsKey;
        // Dispose old instruction sets only after the new key has been rendered.
        this.flushPendingTextInstructions_();
      }
      if (this.textRenderThrottleAuto_) {
        const duration = nowMs() - renderStart;
        this.textRenderDurationAvgMs_ = this.textRenderDurationAvgMs_
          ? this.textRenderDurationAvgMs_ *
              (1 - TEXT_RENDER_THROTTLE_SMOOTHING) +
            duration * TEXT_RENDER_THROTTLE_SMOOTHING
          : duration;
        const viewMoving =
          frameState.viewHints[ViewHint.ANIMATING] ||
          frameState.viewHints[ViewHint.INTERACTING];
        this.updateTextRenderThrottle_(currentFps, viewMoving);
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
    const useOverlayText = !this.gpuStaticLabelReady_;
    // Throttled text refresh during animations (if enabled).
    if (useOverlayText) {
      this.maybeRebuildTextInstructions_(frameState, currentFps);
    }

    // draw the normal canvas
    this.helper.prepareDraw(frameState);
    this.renderWorlds(frameState, false, startWorld, endWorld, worldWidth);

    if (useOverlayText) {
      this.maybeFinalizeTextRender_(frameState, currentFps);
    }

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
    if (
      viewNotMoving &&
      this.gpuStaticLabelRefreshPending_ &&
      this.gpuStaticLabelSoftMode_ &&
      !this.bufferGenerationInFlight_ &&
      !this.isPointAnimationActive_()
    ) {
      this.refreshGpuStaticLabels_();
    }
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
    this.releaseGpuStaticLabelAtlases_();
    super.disposeInternal();
  }

  renderDeclutter() {}
}

export default WebGLVectorLayerRenderer;

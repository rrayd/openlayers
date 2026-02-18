/**
 * A worker that does cpu-heavy tasks related to webgl rendering.
 * @module ol/worker/textOverlay
 */

import {newParsingContext} from '../expr/expression.js';
import Executor from '../render/canvas/Executor.js';
import TextBuilder from '../render/canvas/TextBuilder.js';
import {flatStyleLikeToStyleFunction} from '../render/canvas/style.js';
import {TextOverlayWorkerMessageType} from '../render/webgl/constants.js';
import {deserializeFrameState} from '../render/webgl/serialize.js';
import {
  convertLineStringRenderInstructionsToCanvasTextBuilder,
  convertPointRenderInstructionsToCanvasTextBuilder,
  convertPolygonRenderInstructionsToCanvasTextBuilder,
  stripNonTextStyleProperties,
} from '../render/webgl/textUtil.js';
import {
  compose as composeTransform,
  create as createTransform,
  invert as invertTransform,
  multiply as multiplyTransform,
} from '../transform.js';

/** @type {any} */
const worker = self;

let textRenderAnimationFrameKey = 0;
let textRenderInProgress = false;
const pendingRenders = [];
let lastNonEmptyRenderBatchKeys = [];
let instructionsSetCounter = 0;

const DEBUG_TEXT_OVERLAY = false;
const DEBUG_TEXT_OVERLAY_LOG_EVERY = 60;
let debugTextOverlayRenderCount = 0;

const canvas = new OffscreenCanvas(1, 1);
const context = canvas.getContext('2d');

/**
 * @typedef {Object} RenderBatch
 * @property {import('../transform.js').Transform} inverseTransform Inverse of transform in which coordinates are expressed
 * @property {import('../render/canvas/Executor.js').default} executor Executor containing instructions
 */

/**
 * This holds all constructed render batches
 * If the value is null, it means the render batch has been processed but no text instruction was needed there
 * @type {Map<string, RenderBatch|null>}
 */
const renderBatches = new Map();

/**
 * This holds the ids of the render batches to render in the next frame; will be cleared after each frame
 * @type {Set<string>}
 */
const renderBatchList = new Set();

const tmpTransform = createTransform();

function getRenderTransform(
  center,
  resolution,
  rotation,
  pixelRatio,
  width,
  height,
  offsetX,
) {
  const dx1 = width / 2;
  const dy1 = height / 2;
  const sx = pixelRatio / resolution;
  const sy = -sx;
  const dx2 = -center[0] + offsetX;
  const dy2 = -center[1];
  return composeTransform(tmpTransform, dx1, dy1, sx, sy, -rotation, dx2, dy2);
}

function scheduleTextRender() {
  if (
    textRenderAnimationFrameKey ||
    textRenderInProgress ||
    !pendingRenders.length
  ) {
    return;
  }

  textRenderAnimationFrameKey = requestAnimationFrame(() => {
    textRenderAnimationFrameKey = 0;
    textRenderInProgress = true;

    const renderJob = pendingRenders.shift();
    if (!renderJob) {
      textRenderInProgress = false;
      scheduleTextRender();
      return;
    }

    const {id, frameState: frameStateSerialized, renderBatchKeys} = renderJob;
    const frameState = deserializeFrameState(frameStateSerialized);
    const viewState = frameState.viewState;

    const renderStart = DEBUG_TEXT_OVERLAY ? performance.now() : 0;

    // either resize or clear
    if (
      frameState.size[0] !== canvas.width ||
      frameState.size[1] !== canvas.height
    ) {
      canvas.width = frameState.size[0];
      canvas.height = frameState.size[1];
    } else {
      context.clearRect(0, 0, canvas.width, canvas.height);
    }

    for (const renderBatchKey of renderBatchKeys) {
      if (!renderBatches.has(renderBatchKey)) {
        // eslint-disable-next-line no-console
        console.warn('Unknown render batch key ', renderBatchKey); // TODO: this should not happen, maybe throw here?
        continue;
      }
      const renderBatch = renderBatches.get(renderBatchKey);
      if (!renderBatch) {
        // no instructions there
        continue;
      }
      const transform = getRenderTransform(
        viewState.center,
        viewState.resolution,
        0,
        frameState.pixelRatio,
        canvas.width,
        canvas.height,
        0,
      );
      multiplyTransform(transform, renderBatch.inverseTransform);

      renderBatch.executor.execute(
        context,
        frameState.size,
        transform,
        frameState.viewState.rotation,
        false,
      );
    }

    const sendResponse = (imageData, id) => {
      /** @type {import('../render/webgl/constants.js').TextOverlayWorkerMessage} */
      const message = {
        type: TextOverlayWorkerMessageType.RENDER,
        imageData,
        frameState: frameStateSerialized,
        id,
      };
      worker.postMessage(message, [imageData]);
    };

    sendResponse(canvas.transferToImageBitmap(), id);
    textRenderInProgress = false;
    if (DEBUG_TEXT_OVERLAY) {
      debugTextOverlayRenderCount++;
      if (debugTextOverlayRenderCount % DEBUG_TEXT_OVERLAY_LOG_EVERY === 0) {
        const renderTime = performance.now() - renderStart;
        // eslint-disable-next-line no-console
        console.debug('textOverlay render', {
          ms: Math.round(renderTime),
          batches: renderBatchKeys.length,
          pending: pendingRenders.length,
        });
      }
    }
    scheduleTextRender();
  });
}

function enqueueTextRender(id, frameStateSerialized) {
  // Snapshot the current render list so new entries can be queued for the next frame.
  const snapshotBatchKeys = Array.from(renderBatchList.values());
  renderBatchList.clear();
  let renderBatchKeys = snapshotBatchKeys.filter((key) =>
    renderBatches.has(key),
  );

  // If this frame raced and did not receive ADD_TO_RENDER_LIST yet, reuse the last
  // non-empty set to avoid transient empty overlays (visible flicker).
  if (!renderBatchKeys.length && snapshotBatchKeys.length === 0) {
    renderBatchKeys = lastNonEmptyRenderBatchKeys.filter((key) =>
      renderBatches.has(key),
    );
  }
  if (renderBatchKeys.length) {
    lastNonEmptyRenderBatchKeys = renderBatchKeys.slice();
  }

  const renderJob = {
    id,
    frameState: frameStateSerialized,
    renderBatchKeys,
  };
  if (pendingRenders.length) {
    // Keep only the latest request to avoid rendering stale intermediate frames.
    pendingRenders[pendingRenders.length - 1] = renderJob;
  } else {
    pendingRenders.push(renderJob);
  }
  if (DEBUG_TEXT_OVERLAY && pendingRenders.length > 1) {
    // eslint-disable-next-line no-console
    console.debug('textOverlay render queue', pendingRenders.length);
  }
  scheduleTextRender();
}

worker.onmessage = (event) => {
  const received = event.data;
  switch (received.type) {
    case TextOverlayWorkerMessageType.ADD_TO_RENDER_LIST: {
      const {instructionsSetKey} = received;
      renderBatchList.add(instructionsSetKey);
      break;
    }

    case TextOverlayWorkerMessageType.RENDER: {
      enqueueTextRender(received.id, received.frameState);
      break;
    }

    case TextOverlayWorkerMessageType.BUILD_INSTRUCTIONS: {
      if (DEBUG_TEXT_OVERLAY) {
        // eslint-disable-next-line no-console
        console.time('BUILD_INSTRUCTIONS');
      }
      const {
        polygonRenderInstructions,
        lineStringRenderInstructions,
        pointRenderInstructions,
        style,
        customAttributesSizes,
        renderInstructionsTransform,
        buildClipExtent,
        id,
        // TODO: view projection
      } = received;
      const resolution = 1;
      const pixelRatio = 1;
      const maxExtent =
        buildClipExtent && buildClipExtent.length === 4
          ? buildClipExtent
          : [-Infinity, -Infinity, Infinity, Infinity];
      const instructionsSetKey = String(++instructionsSetCounter);
      const labelsArray = new Uint8Array(received.labelsArray);
      const builder = new TextBuilder(1, maxExtent, resolution, pixelRatio);

      const parsingContext = newParsingContext();
      stripNonTextStyleProperties(style);
      const styleFn = flatStyleLikeToStyleFunction(style, parsingContext);

      if (DEBUG_TEXT_OVERLAY) {
        // eslint-disable-next-line no-console
        console.timeLog('BUILD_INSTRUCTIONS', '/ converted style to styleFn');
      }

      convertPolygonRenderInstructionsToCanvasTextBuilder(
        new Float32Array(polygonRenderInstructions),
        labelsArray,
        parsingContext.properties,
        customAttributesSizes,
        builder,
        styleFn,
      );
      if (DEBUG_TEXT_OVERLAY) {
        // eslint-disable-next-line no-console
        console.timeLog('BUILD_INSTRUCTIONS', '/ parsed polygon instructions');
      }
      convertLineStringRenderInstructionsToCanvasTextBuilder(
        new Float32Array(lineStringRenderInstructions),
        renderInstructionsTransform,
        labelsArray,
        parsingContext.properties,
        customAttributesSizes,
        builder,
        styleFn,
      );
      if (DEBUG_TEXT_OVERLAY) {
        // eslint-disable-next-line no-console
        console.timeLog(
          'BUILD_INSTRUCTIONS',
          '/ parsed lineString instructions',
        );
      }
      convertPointRenderInstructionsToCanvasTextBuilder(
        new Float32Array(pointRenderInstructions),
        renderInstructionsTransform,
        labelsArray,
        parsingContext.properties,
        customAttributesSizes,
        builder,
        styleFn,
        buildClipExtent,
      );
      if (DEBUG_TEXT_OVERLAY) {
        // eslint-disable-next-line no-console
        console.timeLog('BUILD_INSTRUCTIONS', '/ parsed point instructions');
      }

      const canvasInstructions = builder.finish();

      // nothing to draw
      if (canvasInstructions.instructions.length === 0) {
        renderBatches.set(instructionsSetKey, null);
      } else {
        const inverseTransform = invertTransform(renderInstructionsTransform);
        const executor = new Executor(
          resolution,
          pixelRatio,
          false,
          canvasInstructions,
        );
        renderBatches.set(instructionsSetKey, {
          inverseTransform,
          executor,
        });
      }

      /** @type {import('../render/webgl/constants.js').TextOverlayWorkerMessage} */
      const message = {
        type: TextOverlayWorkerMessageType.BUILD_INSTRUCTIONS,
        instructionsSetKey,
        id,
      };
      worker.postMessage(message);

      if (DEBUG_TEXT_OVERLAY) {
        // eslint-disable-next-line no-console
        console.timeEnd('BUILD_INSTRUCTIONS');
      }
      break;
    }

    case TextOverlayWorkerMessageType.DISPOSE_INSTRUCTIONS: {
      const {instructionsSetKey} = received;
      if (renderBatches.has(instructionsSetKey)) {
        renderBatches.delete(instructionsSetKey);
      }
      renderBatchList.delete(instructionsSetKey);
      if (lastNonEmptyRenderBatchKeys.length) {
        lastNonEmptyRenderBatchKeys = lastNonEmptyRenderBatchKeys.filter(
          (key) => key !== instructionsSetKey,
        );
      }
      break;
    }

    default:
    // pass
  }
};

/** @type {function(): Worker} */ export let create;

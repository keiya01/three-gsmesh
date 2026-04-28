import {
  Color,
  Mesh,
  PerspectiveCamera,
  Quaternion,
  RenderPipeline,
  Scene,
  Vector3,
  WebGPURenderer,
} from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { acesFilmicToneMapping, pass, uniform, vec4, vibrance } from 'three/tsl';
import { sharpen } from 'three/examples/jsm/tsl/display/SharpenNode.js';

import './style.css';

import { GSMesh, PlyLoader, SpzLoader, type SplatData } from 'three-gsmesh';
import { PaneUI, type PaneUIState } from './pane';

const PRESETS = {
  'Cherry Blossoms': '/CHERRY BLOSSOMS/scene.spz',
  'Little Plant': '/Little-Plant/scene.spz',
  Flowers: '/Flowers/scene.spz',
} as const;

type PresetName = keyof typeof PRESETS;

const PRESET_CREDITS: Record<PresetName, {
  title: string;
  author: string;
  sourceUrl: string;
  license: string;
  licenseUrl: string;
}> = {
  'Cherry Blossoms': {
    title: 'Cherry Blossoms',
    author: 'Todd Smith',
    sourceUrl: 'https://superspl.at/scene/4fe67b3d',
    license: 'CC BY-NC-SA 4.0',
    licenseUrl: 'http://creativecommons.org/licenses/by-nc-sa/4.0/',
  },
  'Little Plant': {
    title: 'Little Plant',
    author: 'Pierrick',
    sourceUrl: 'https://superspl.at/scene/c5af35a1',
    license: 'CC BY 4.0',
    licenseUrl: 'http://creativecommons.org/licenses/by/4.0/',
  },
  Flowers: {
    title: 'Flowers',
    author: 'Oscar F',
    sourceUrl: 'https://superspl.at/scene/7f8a7674',
    license: 'CC BY 4.0',
    licenseUrl: 'http://creativecommons.org/licenses/by/4.0/',
  },
};

const isPresetName = (value: string): value is PresetName => value in PRESETS;

const IDENTITY_QUATERNION = new Quaternion();
const FLIP_Y_QUATERNION = new Quaternion().setFromAxisAngle(
  new Vector3(1, 0, 0),
  Math.PI,
);

const getLoader = (url: string, includeSH: boolean) => {
  const extension = url.split('?')[0].split('#')[0].toLowerCase().split('.').pop();

  switch (extension) {
    case 'ply':
      return new PlyLoader({ includeSH });
    case 'spz':
      return new SpzLoader({ includeSH });
    default:
      throw new Error(`Unsupported file type: ${extension ?? url}`);
  }
};

const disposeMesh = (mesh: GSMesh) => {
  mesh.traverse((child) => {
    if (!(child instanceof Mesh)) return;

    const object = child;

    object.geometry?.dispose();

    if (Array.isArray(object.material)) {
      object.material.forEach((material) => material.dispose());
    } else {
      object.material?.dispose();
    }
  });
};

const resolveSource = (state: PaneUIState) => {
  if (state.file) {
    return {
      key: `file:${state.file.name}:${state.file.size}:${state.file.lastModified}`,
      label: state.file.name,
      file: state.file,
      url: state.file.name,
    };
  }

  if (!isPresetName(state.data)) {
    return null;
  }

  return {
    key: `preset:${state.data}`,
    label: state.data,
    file: null,
    url: PRESETS[state.data],
  };
};

type SplatSource = NonNullable<ReturnType<typeof resolveSource>>;

const run = async () => {
  const app = document.querySelector<HTMLDivElement>('#app');
  
  if (!app) {
    throw new Error('Missing #app element');
  }

  const uiContainer = document.querySelector<HTMLElement>('#ui') ?? document.createElement('div');
  uiContainer.id = 'ui';
  document.body.appendChild(uiContainer);

  const credit = document.createElement('div');
  credit.id = 'credit';
  credit.hidden = true;
  document.body.appendChild(credit);

  const renderer = new WebGPURenderer({});
  // renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  app.appendChild(renderer.domElement);
  
  const scene = new Scene();
  scene.background = new Color("#558fa5");
  
  const camera = new PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    1000,
  );
  camera.position.z = 6;
  camera.position.y = 2.5;

  const renderInfo = {
    requested: true,
  };
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.copy(scene.position);
  controls.addEventListener('change', () => {
    renderInfo.requested = true;
  });
  controls.update();

  // Post-processing
  const renderPipeline = new RenderPipeline(renderer);
  const scenePass = pass(scene, camera);
  const vibranceValue = uniform(0.2);
  const exposureValue = uniform(1);
  const adjustedColor = vibrance(scenePass, vibranceValue);
  renderPipeline.outputNode = acesFilmicToneMapping(
    sharpen(vec4(adjustedColor, scenePass.a), 0),
    exposureValue,
  );
  
  let activeMesh: GSMesh | null = null;
  let activeMeshKey: string | null = null;
  const splatDataCache = new Map<string, Promise<SplatData>>();
  let loadVersion = 0;

  const applyMeshFlip = (flipY: boolean) => {
    activeMesh?.quaternion.copy(flipY ? FLIP_Y_QUATERNION : IDENTITY_QUATERNION);
    renderInfo.requested = true;
  };

  const loadSplatData = (
    source: SplatSource,
    includeSH: boolean,
    dataKey: string,
  ) => {
    const cachedData = splatDataCache.get(dataKey);

    if (cachedData) {
      return cachedData;
    }

    const loader = getLoader(source.url, includeSH);
    const dataPromise = source.file
      ? source.file.arrayBuffer().then((buffer) => loader.parseData(buffer))
      : loader.loadAsync(source.url);

    splatDataCache.set(dataKey, dataPromise);

    dataPromise.catch(() => {
      if (splatDataCache.get(dataKey) === dataPromise) {
        splatDataCache.delete(dataKey);
      }
    });

    return dataPromise;
  };

  const loadSplat = async (state: PaneUIState) => {
    const source = resolveSource(state);
    if (!source) {
      return;
    }

    const dataKey = [
      source.key,
      `includeSH:${state.includeSH}`,
    ].join('|');
    const meshKey = [
      dataKey,
      `renderMode:${state.renderMode}`,
      `showEllipsoid:${state.showEllipsoid}`,
    ].join('|');

    if (meshKey === activeMeshKey) {
      return;
    }

    const currentLoad = ++loadVersion;

    try {
      const data = await loadSplatData(source, state.includeSH, dataKey);

      if (currentLoad !== loadVersion) {
        return;
      }

      if (activeMesh) {
        scene.remove(activeMesh);
        disposeMesh(activeMesh);
      }

      activeMesh = new GSMesh(data, {
        renderMode: state.renderMode,
        showEllipsoid: state.showEllipsoid,
      });
      activeMeshKey = meshKey;
      applyMeshFlip(state.flipY);
      scene.add(activeMesh);
    } catch (error) {
      if (currentLoad === loadVersion) {
        activeMeshKey = null;
      }

      console.error(`Failed to load ${source.label}:`, error);
    }
  };

  const updateCredit = (state: PaneUIState) => {
    const presetName = !state.file && isPresetName(state.data) ? state.data : null;

    credit.replaceChildren();

    if (!presetName) {
      credit.hidden = true;
      return;
    }

    const presetCredit = PRESET_CREDITS[presetName];
    const title = document.createElement('a');
    title.href = presetCredit.sourceUrl;
    title.target = '_blank';
    title.rel = 'noreferrer';
    title.textContent = presetCredit.title;

    const license = document.createElement('a');
    license.href = presetCredit.licenseUrl;
    license.target = '_blank';
    license.rel = 'noreferrer';
    license.textContent = presetCredit.license;

    credit.append(
      title,
      document.createTextNode(` by ${presetCredit.author} · `),
      license,
    );
    credit.hidden = false;
  };

  const applyState = async (state: PaneUIState) => {
    scene.background = new Color(state.background);
    vibranceValue.value = state.vibrance;
    exposureValue.value = 2 ** state.exposure;

    updateCredit(state);
    applyMeshFlip(state.flipY);

    await loadSplat(state);

    renderInfo.requested = true;
  };

  const ui = new PaneUI({
    container: uiContainer,
    dataOptions: Object.keys(PRESETS),
    accept: '.ply,.spz',
    initialState: {
      data: 'Cherry Blossoms',
      renderMode: 'billboard',
      showEllipsoid: false,
      includeSH: true,
      background: '#558fa5',
      vibrance: 0.2,
      exposure: 0,
      flipY: true,
    },
  });

  ui.onChange((state) => {
    applyState(state);
  });

  renderer.setAnimationLoop(() => {
    if(!renderInfo.requested) return;
    renderer.render(scene, camera);
    renderPipeline.render();
    renderInfo.requested = controls.update();
  });

  window.addEventListener('resize', () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
  
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  
    renderer.setSize(width, height);
    renderInfo.requested = true;
  });
}

run();

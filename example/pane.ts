import { Pane } from "tweakpane";

export type RenderMode = "point" | "billboard";

export type PaneUIState = {
  data: string;
  file: File | null;
  fileUrl: string | null;
  renderMode: RenderMode;
  showEllipsoid: boolean;
  includeSH: boolean;
  background: string;
  vibrance: number;
  exposure: number;
  flipY: boolean;
};

type PaneUIChangeHandler = (state: PaneUIState) => void;

export class PaneUI {
  private pane: Pane;
  private state: PaneUIState;
  private listeners = new Set<PaneUIChangeHandler>();

  private root: HTMLElement;
  private fileInput: HTMLInputElement;
  private currentObjectUrl: string | null = null;

  constructor(options: {
    container?: HTMLElement;
    dataOptions: string[];
    initialState?: Partial<PaneUIState>;
    accept?: string;
  }) {
    const {
      container,
      dataOptions,
      initialState,
      accept = "*/*",
    } = options;

    this.root = document.createElement("div");
    container?.appendChild(this.root);

    this.state = {
      data: dataOptions[0] ?? "",
      file: null,
      fileUrl: null,
      renderMode: "billboard",
      showEllipsoid: false,
      includeSH: true,
      background: "#000000",
      vibrance: 0,
      exposure: 0,
      flipY: false,
      ...initialState,
    };

    this.pane = new Pane({
      title: "Controls",
      container: this.root,
    });

    this.fileInput = this.createFileInput(accept);

    this.build(dataOptions);
  }

  private build(dataOptions: string[]) {
    this.pane
      .addBinding(this.state, "data", {
        label: "Data",
        options: Object.fromEntries(
          dataOptions.map((value) => [value, value]),
        ),
      })
      .on("change", () => {
        this.clearFile(false);
        this.emit();
      });

    this.pane
      .addBinding(this.state, "renderMode", {
        label: "Render Mode",
        options: {
          Billboard: "billboard",
          Point: "point",
        },
      })
      .on("change", () => this.emit());

    this.pane
      .addBinding(this.state, "showEllipsoid", {
        label: "Show Ellipsoid",
      })
      .on("change", () => this.emit());

    this.pane
      .addBinding(this.state, "includeSH", {
        label: "Add SH",
      })
      .on("change", () => this.emit());

    this.pane
      .addBinding(this.state, "background", {
        label: "Background",
        view: "color",
      })
      .on("change", () => this.emit());

    this.pane
      .addBinding(this.state, "vibrance", {
        label: "Vibrance",
        min: -1,
        max: 1,
        step: 0.01,
      })
      .on("change", () => this.emit());

    this.pane
      .addBinding(this.state, "exposure", {
        label: "Exposure",
        min: -5,
        max: 5,
        step: 0.01,
      })
      .on("change", () => this.emit());

    this.pane
      .addBinding(this.state, "flipY", {
        label: "Flip Y",
      })
      .on("change", () => this.emit());

    this.root.appendChild(this.fileInput);
  }

  private createFileInput(accept: string): HTMLInputElement {
    const input = document.createElement("input");

    input.type = "file";
    input.accept = accept;
    input.style.display = "block";
    input.style.marginTop = "8px";
    input.style.width = "100%";
    input.style.fontWeight = "bold";

    input.addEventListener("change", () => {
      const file = input.files?.[0] ?? null;

      if (!file) {
        this.clearFile();
        return;
      }

      this.setFile(file);
    });

    return input;
  }

  private setFile(file: File) {
    if (this.currentObjectUrl) {
      URL.revokeObjectURL(this.currentObjectUrl);
    }

    const fileUrl = URL.createObjectURL(file);
    this.currentObjectUrl = fileUrl;

    this.state.file = file;
    this.state.fileUrl = fileUrl;
    this.state.data = file.name;

    this.pane.refresh();
    this.emit();
  }

  private clearFile(emit = true) {
    if (this.currentObjectUrl) {
      URL.revokeObjectURL(this.currentObjectUrl);
      this.currentObjectUrl = null;
    }

    this.state.file = null;
    this.state.fileUrl = null;

    this.fileInput.value = "";

    if (emit) {
      this.emit();
    }
  }

  onChange(handler: PaneUIChangeHandler) {
    this.listeners.add(handler);
    handler(this.getState());

    return () => {
      this.listeners.delete(handler);
    };
  }

  getState(): PaneUIState {
    return {
      ...this.state,
    };
  }

  setState(nextState: Partial<PaneUIState>) {
    if (nextState.file) {
      this.setFile(nextState.file);
      return;
    }

    this.state = {
      ...this.state,
      ...nextState,
    };

    if (nextState.file === null || nextState.fileUrl === null) {
      this.clearFile(false);
    }

    this.pane.refresh();
    this.emit();
  }

  dispose() {
    this.clearFile(false);
    this.listeners.clear();
    this.pane.dispose();
    this.root.remove();
  }

  private emit() {
    const snapshot = this.getState();

    for (const listener of this.listeners) {
      listener(snapshot);
    }
  }
}

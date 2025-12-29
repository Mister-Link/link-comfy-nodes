export class ModelAPI {
  static async listModels() {
    try {
      const resp = await fetch("/workflow_checker/list_models");
      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        return { error: errJson.reason || `HTTP ${resp.status}`, path: errJson.path };
      }
      const json = await resp.json();
      return json.ok && Array.isArray(json.data) ? { data: json.data, path: json.path } : { error: "Invalid response" };
    } catch (err) {
      return { error: err.message };
    }
  }

  static getErrorMessage(error, path) {
    const messages = {
      missing_dir: `~/.config/comfy/ folder missing. Create it and add models.json at ${path}`,
      missing_file: `models.json missing. Expected at ${path}`,
      invalid_format: "Invalid models.json format. Expected an array of model entries.",
    };
    return messages[error] || error;
  }
}
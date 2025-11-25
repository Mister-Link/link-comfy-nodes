import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "Comfy.LinkComfy.NodeWidth",
  async nodeCreated(node) {
    // Set default width for specific nodes
    if (node.comfyClass === "Image Rotator") {
      node.setSize([210, node.size[1]]);
    } else if (node.comfyClass === "Pose Image Setup") {
      node.setSize([210, node.size[1]]);
    }
  },
});

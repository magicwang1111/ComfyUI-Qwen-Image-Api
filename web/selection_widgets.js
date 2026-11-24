import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Goumei.SelectionWidgets",
    beforeConfigureGraph(graphData) {
        for (const node of graphData.nodes ?? []) {
            if (node.type !== "GoumeiSelectLogoDetails") continue;
            node.widgets_values = [];
            delete node.widgets_values_named;
        }
    },
});

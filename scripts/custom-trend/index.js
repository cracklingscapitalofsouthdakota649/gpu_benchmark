#scripts/custom-trend/index.js
AllurePluginApi.addPlugin("custom-trend", function (context) {
    const trendItems = context.trendItems || [];
    const gpuUtilTrend = [];
    const gpuMemTrend = [];
    const cpuUtilTrend = [];
    const fpsTrend = [];

    trendItems.forEach(item => {
        if (item.gpu_avg_util !== undefined) {
            gpuUtilTrend.push({ buildOrder: item.buildOrder, value: item.gpu_avg_util, name: "GPU Utilization (%)", color: "#00C853" });
        }
        if (item.gpu_avg_mem !== undefined) {
            gpuMemTrend.push({ buildOrder: item.buildOrder, value: item.gpu_avg_mem, name: "VRAM (MB)", color: "#2979FF" });
        }
        if (item.cpu_avg_util !== undefined) {
            cpuUtilTrend.push({ buildOrder: item.buildOrder, value: item.cpu_avg_util, name: "CPU Utilization (%)", color: "#FF9800" });
        }
        if (item.fps !== undefined) {
            fpsTrend.push({ buildOrder: item.buildOrder, value: item.fps, name: "FPS / Ops/sec", color: "#9C27B0" });
        }
    });

    if (gpuUtilTrend.length) context.addTrend({ name: "GPU Utilization (%)", data: gpuUtilTrend, color: "#00C853", strokeWidth: 3, fillOpacity: 0.1 });
    if (gpuMemTrend.length) context.addTrend({ name: "VRAM (MB)", data: gpuMemTrend, color: "#2979FF", strokeWidth: 3, fillOpacity: 0.1 });
    if (cpuUtilTrend.length) context.addTrend({ name: "CPU Utilization (%)", data: cpuUtilTrend, color: "#FF9800", strokeWidth: 3, fillOpacity: 0.1 });
    if (fpsTrend.length) context.addTrend({ name: "FPS / Ops/sec", data: fpsTrend, color: "#9C27B0", strokeWidth: 3, fillOpacity: 0.1 });
});

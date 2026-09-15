// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

window.addEventListener("DOMContentLoaded", () => {
  const details = document.querySelectorAll(
    ".benchmark-estimator-details details.sd-dropdown",
  );
  document.querySelectorAll("[data-benchmark-details]").forEach((button) => {
    button.addEventListener("click", () => {
      const open = button.dataset.benchmarkDetails === "expand";
      details.forEach((detail) => {
        detail.open = open;
      });
    });
  });
});

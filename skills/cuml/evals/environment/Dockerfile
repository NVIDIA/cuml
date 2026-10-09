# Eval environment for the cuml skill (NVIDIA SkillEvaluator, rebase mode: the FROM line is
# replaced by SkillEvaluator's python:3.12-slim eval base image; the RUN lines are kept).
#
# Versions are pinned so a rebuild installs the same packages. cuML is the latest release on
# PyPI; scikit-learn, umap-learn and hdbscan are the newest versions inside that release's
# cuml.accel tested window (_CONSTRAINTS in python/cuml/cuml/accel/core.py on release/26.08:
# scikit-learn <=1.9.0, umap-learn <=0.5.12, hdbscan <=0.8.44).
# When bumping cuML, update all four pins together. See
# docs/source/developer_guide/agent_skill.md.
FROM python:3.12-slim

# GPU access: only takes effect when Docker's default runtime is nvidia (see the developer
# guide). Set the device to a GPU on the evaluation machine (`nvidia-smi -L`); without the
# nvidia default runtime this line is ignored and trials run without a GPU.
ENV NVIDIA_VISIBLE_DEVICES=GPU-070d24ec-8d79-30fd-3bf7-405d86ddb212 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN pip install --no-cache-dir \
    cuml-cu13==26.8.0 \
    scikit-learn==1.9.0 \
    umap-learn==0.5.12 \
    hdbscan==0.8.44

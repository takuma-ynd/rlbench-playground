#!/usr/bin/env bash
. /share/data/ripl/takuma/lib/miniconda3/etc/profile.d/conda.sh
export PATH=/share/data/ripl/takuma/lib/miniconda3/condabin:/share/data/ripl/takuma/lib/miniconda3/bin:$PATH
conda deactivate
conda activate rlbench
export COPPELIASIM_ROOT=/workspace/CoppeliaSim_Edu_V4_0_0_Ubuntu16_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:0

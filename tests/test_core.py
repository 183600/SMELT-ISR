"""Aggregated fast regression tests (split into cohesive modules)."""

from _core_support import *  # noqa: F401,F403

from _core_CoreModelTests import CoreModelTests
from _core_CostModelTests import CostModelTests
from _core_StatisticsTests import StatisticsTests
from _core_StatisticsTests import ResultsAnalysisDiagnosticsTests
from _core_DesignPreparationTests_00 import DesignPreparationTests
from _core_DesignPreparationTests_01 import DesignPreparationTestsPart2
from _core_DesignPreparationTests_02 import DesignPreparationTestsPart3
from _core_DesignPreparationTests_base import _DesignPreparationTestsBase
from _core_DesignPreflightTests import DesignPreflightTests
from _core_RoutingTests import RoutingTests
from _core_ReproducibilityTests import ReproducibilityTests
from _core_RunAuditTests import RunAuditTests
from _core_NearDuplicateDataTests import NearDuplicateDataTests
from _core_DataPlanTests import DataPlanTests
from _core_ProcessTests import ProcessTests
from _core_PreregistrationTests import PreregistrationTests
from _core_SchemaTests import SchemaStructureTests, ExampleConsistencyTests
from _core_ArtifactManifestTests import ArtifactManifestTests
from _core_EnvironmentSnapshotTests import EnvironmentSnapshotTests
from _core_EdgeCaseRegressionTests_00 import EdgeCaseRegressionTests
from _core_EdgeCaseRegressionTests_01 import EdgeCaseRegressionTestsPart2
from _core_NumericalScaleTests import NumericalScaleTests
from _core_CurveMeanMetricTests import CurveMeanMetricTests
from _core_GroupClusterRegressionTests import GroupClusterRegressionTests
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()

__all__ = [
    "CoreModelTests",
    "CostModelTests",
    "StatisticsTests",
    "ResultsAnalysisDiagnosticsTests",
    "DesignPreparationTests",
    "DesignPreparationTestsPart2",
    "DesignPreparationTestsPart3",
    "_DesignPreparationTestsBase",
    "DesignPreflightTests",
    "RoutingTests",
    "ReproducibilityTests",
    "RunAuditTests",
    "NearDuplicateDataTests",
    "DataPlanTests",
    "ProcessTests",
    "PreregistrationTests",
    "SchemaStructureTests",
    "ExampleConsistencyTests",
    "ArtifactManifestTests",
    "EnvironmentSnapshotTests",
    "EdgeCaseRegressionTests",
    "EdgeCaseRegressionTestsPart2",
    "NumericalScaleTests",
    "CurveMeanMetricTests",
    "GroupClusterRegressionTests",
]

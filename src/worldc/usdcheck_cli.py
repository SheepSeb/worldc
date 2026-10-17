"""usdchecker (pxr.UsdUtils.complianceChecker) as JSON.  Needs a USD build that ships shader resources,
i.e. Isaac Sim's ``omni.usd.libs`` (the pip ``usd-core`` wheel lacks them).  Invoked by run_all.py."""
import json
import sys

from pxr.UsdUtils import complianceChecker as cc

chk = cc.ComplianceChecker(arkit=False, skipARKitRootLayerCheck=False, rootPackageOnly=False, skipVariants=False, verbose=False)
chk.CheckCompliance(sys.argv[1])
print("JSON:" + json.dumps({"errors": chk.GetErrors(), "failed_checks": chk.GetFailedChecks(), "warnings": chk.GetWarnings()}))

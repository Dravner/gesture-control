"""Mark interactive camera work as user initiated without permanent OS settings."""
import sys

class CameraActivity:
    def __init__(self,platform=None,provider=None):
        self.platform=sys.platform if platform is None else platform
        self.provider=provider;self._process=None;self._token=None
    @property
    def active(self):return self._token is not None
    def start(self):
        if self.active:return True
        if self.platform!='darwin':return False
        if self.provider is not None:process,options=self.provider()
        else:
            try:import Foundation as f
            except ImportError:return False
            process=f.NSProcessInfo.processInfo();options=f.NSActivityUserInitiatedAllowingIdleSystemSleep
        token=process.beginActivityWithOptions_reason_(options,'Interactive gesture-control camera session')
        self._process=process;self._token=token
        return self.active
    def stop(self):
        process,token=self._process,self._token;self._process=None;self._token=None
        if token is not None:process.endActivity_(token)

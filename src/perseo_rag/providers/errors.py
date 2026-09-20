class ProviderError(RuntimeError):
    pass


class ProviderTimeout(ProviderError):
    pass


class ProviderUnavailable(ProviderError):
    pass


class ProviderRateLimited(ProviderError):
    pass


class ProviderProtocolError(ProviderError):
    pass

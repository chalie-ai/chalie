"""Response DTO for the provider credential-reveal action.

The one provider read shape that deliberately carries the stored credential back
to the admin. Every other shape in this group strips it (see
:mod:`api.response.provider`), so the exception lives in its own type rather than
as a nullable field on the ordinary read shape — where a later handler could
return it without meaning to.
"""

from __future__ import annotations

from .response import Response


class ProviderCredential(Response):
    """One provider's stored API key, decrypted for the edit form.

    ``api_key`` is ``None`` both for a row that stores no credential and for one
    the vault cannot currently decrypt. The form treats them the same way — an
    empty field the operator can type into — so they are not distinguished here.
    """

    id: int
    api_key: str | None

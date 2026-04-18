"""
DataGateway datacube contract example showing validator rejection handling.
"""

from __future__ import annotations

import pandas as pd

from boti_data.datacube import DatacubeConfig, DatacubeContract
from boti_data.gateway import DataGateway


def _loader(_request):
    return pd.DataFrame({"id": [1], "status": ["active"]})


def _validator(request) -> None:
    if request.cube != "orders":
        raise ValueError("cube must be 'orders'")


def main() -> dict[str, str]:
    contract = DatacubeContract(loader=_loader, request_validator=_validator)
    config = DatacubeConfig(contract=contract)

    with DataGateway(config) as gateway:
        try:
            gateway.load(return_type="pandas", cube="inventory", status__exact="active")
        except ValueError as exc:
            message = str(exc)
            print("Expected contract validation error:")
            print(message)
            return {"error": message}

    raise RuntimeError("Expected contract validation error was not raised.")


if __name__ == "__main__":
    main()


# src/models/factory.py 
from src.models.halccon import HALCCONMulticlass
from src.models.loss_eqlv2 import EQLv2Loss


def build_model(model_config: dict):
    model_name = model_config["model_name"]

    if model_name in ["halccon", "halccon_multiclass_ieee_baseline"]:
        input_dim = model_config["architecture"]["input_dim"]
        num_classes = model_config["architecture"]["num_classes"]

        if input_dim is None:
            raise ValueError("model_config['architecture']['input_dim'] no puede ser None.")
        if num_classes is None:
            raise ValueError("model_config['architecture']['num_classes'] no puede ser None.")

        return HALCCONMulticlass(
            input_dim=input_dim,
            num_classes=num_classes,
        )

    raise ValueError(f"Modelo no soportado: {model_name}")


def build_loss(loss_config: dict, num_classes: int):
    loss_name = loss_config["name"].lower()

    if loss_name == "eqlv2":
        params = loss_config.get("params", {})
        return EQLv2Loss(
            num_classes=num_classes,
            gamma=params.get("gamma", 12.0),
            mu=params.get("mu", 0.3),
            alpha=params.get("alpha", 2.0),
            loss_weight=params.get("loss_weight", 1.0),
            vis_grad=params.get("vis_grad", False),
        )

    raise ValueError(f"Función de pérdida no soportada: {loss_name}")
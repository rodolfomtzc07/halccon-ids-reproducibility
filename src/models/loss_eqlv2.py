# src/models/loss_eqlv2.py Este es el módulo de pérdida EQLv2, inspirado en la lógica del notebook. Implementa la función de pérdida con ponderación adaptativa basada en el gradiente acumulado para cada clase, lo que ayuda a manejar el desequilibrio de clases. La función `forward` calcula la pérdida, mientras que `collect_grad` acumula los gradientes para ajustar las ponderaciones en `get_weight`. La función `map_func` aplica una transformación sigmoidal para obtener las ponderaciones finales.
import torch
import torch.nn as nn
import torch.nn.functional as F


class EQLv2Loss(nn.Module):
    """
    EQLv2 alineada al notebook oficial y segura para validación/test.

    Comportamiento:
    - Entrenamiento:
        * actualiza class_counts
        * acumula gradientes positivos/negativos
        * reinicia _pos_grad y _neg_grad por época con on_epoch_start()
    - Evaluación:
        * calcula la pérdida con el estado actual
        * NO modifica class_counts ni gradientes internos
    """

    def __init__(
        self,
        num_classes: int = 13,
        gamma: float = 12.0,
        mu: float = 0.3,
        alpha: float = 2.0,
        loss_weight: float = 1.0,
        eps: float = 1e-10,
        vis_grad: bool = False,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        self.vis_grad = bool(vis_grad)

        self.register_buffer("_pos_grad", torch.zeros(self.num_classes))
        self.register_buffer("_neg_grad", torch.zeros(self.num_classes))
        self.register_buffer("pos_neg", torch.zeros(self.num_classes))
        self.register_buffer("class_counts", torch.zeros(self.num_classes))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: logits [B, C]
        y_true: etiquetas enteras [B]
        """
        if y_pred.ndim != 2:
            raise ValueError(f"y_pred debe tener forma [B, C], recibido: {tuple(y_pred.shape)}")
        if y_true.ndim != 1:
            raise ValueError(f"y_true debe tener forma [B], recibido: {tuple(y_true.shape)}")
        if y_pred.size(1) != self.num_classes:
            raise ValueError(
                f"El número de clases en logits ({y_pred.size(1)}) no coincide con "
                f"num_classes ({self.num_classes})."
            )

        target = y_true.long()
        pred_class_logits = y_pred
        batch_size = pred_class_logits.size(0)

        pos_w, neg_w = self.get_weight(target, update_state=self.training)

        # Alineado a la lógica del notebook:
        # se usa un weight por clase para CrossEntropy
        weight = (pos_w + neg_w).detach()

        cls_loss = F.cross_entropy(
            pred_class_logits,
            target,
            weight=weight,
            reduction="none",
        )
        cls_loss = torch.sum(cls_loss) / batch_size

        # Solo acumular gradientes internos durante entrenamiento
        if self.training:
            self.collect_grad(
                pred_class_logits.detach(),
                target.detach(),
                pos_w.detach(),
                neg_w.detach(),
            )

        return self.loss_weight * cls_loss

    def get_weight(
        self,
        target: torch.Tensor,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Replica la lógica del notebook:
        - class_counts se actualiza con las etiquetas del batch SOLO en entrenamiento
        - class_weights inverso a frecuencia acumulada
        - neg_w = map_func(pos_neg)
        - pos_w = 1 + alpha * (1 - neg_w) * class_weights
        """
        with torch.no_grad():
            if update_state:
                batch_counts = torch.bincount(
                    target, minlength=self.num_classes
                ).to(device=self.class_counts.device, dtype=self.class_counts.dtype)
                self.class_counts += batch_counts

            class_weights = self.class_counts.sum() / (self.class_counts + self.eps)
            class_weights = class_weights / (class_weights.max() + self.eps)

            neg_w = self.map_func(self.pos_neg)
            pos_w = 1.0 + self.alpha * (1.0 - neg_w) * class_weights

        return pos_w.view(-1), neg_w.view(-1)

    def map_func(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-self.gamma * (x - self.mu)))

    def collect_grad(
        self,
        cls_score: torch.Tensor,
        target: torch.Tensor,
        pos_w: torch.Tensor,
        neg_w: torch.Tensor,
    ) -> None:
        """
        Acumula gradientes por clase, alineado con la lógica del notebook.
        Solo debe llamarse durante entrenamiento.
        """
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        prob = torch.sigmoid(cls_score)

        grad = target_one_hot * (prob - 1.0) + (1.0 - target_one_hot) * prob
        grad = torch.abs(grad)

        grad_norm = torch.sum(grad, dim=0)
        grad_norm = grad_norm / (grad_norm.max() + self.eps)

        pos_grad = torch.sum(
            grad * target_one_hot * pos_w.unsqueeze(0), dim=0
        ) / (grad_norm + self.eps)

        neg_grad = torch.sum(
            grad * (1.0 - target_one_hot) * neg_w.unsqueeze(0), dim=0
        ) / (grad_norm + self.eps)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + self.eps)

    def on_epoch_start(self) -> None:
        """
        Igual al notebook: reinicia gradientes acumulados por época.
        class_counts NO se reinicia.
        """
        self._pos_grad.zero_()
        self._neg_grad.zero_()

    def on_epoch_end(self) -> None:
        if self.vis_grad:
            print(f"Gradientes positivos acumulados por clase: {self._pos_grad}")
            print(f"Gradientes negativos acumulados por clase: {self._neg_grad}")
            print(f"Relación pos/neg por clase: {self.pos_neg}")
            print(f"Frecuencia acumulada por clase: {self.class_counts}")

    def reset_all_statistics(self) -> None:
        """
        Útil para Optuna o múltiples runs en el mismo proceso.
        """
        self._pos_grad.zero_()
        self._neg_grad.zero_()
        self.pos_neg.zero_()
        self.class_counts.zero_()

    def get_hparams(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "gamma": self.gamma,
            "mu": self.mu,
            "alpha": self.alpha,
            "loss_weight": self.loss_weight,
            "eps": self.eps,
            "vis_grad": self.vis_grad,
        }
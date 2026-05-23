"""
Knowledge Distillation loss for the Student Depression screening project.

The student is trained against a combined target ``y_true = [y_hard, y_soft]``:
    - ``y_hard``: the ground-truth binary label (0 / 1)
    - ``y_soft``: the teacher model's sigmoid probability ("dark knowledge")

The loss blends a standard Binary Cross-Entropy term (hard targets) with a
temperature-scaled KL-Divergence term (soft targets), following Hinton et al.,
"Distilling the Knowledge in a Neural Network" (2015).

Best configuration found in this project: T = 10.0, alpha = 0.1 (16-8-1 student).
"""

import tensorflow as tf

# --- Default KD hyper-parameters (best combination found) ---
ALPHA = 0.1        # weight on the soft (teacher) loss; (1 - alpha) on the hard loss
TEMPERATURE = 10.0 # softening temperature for the teacher/student logits


def distillation_loss(y_true, y_pred, alpha=ALPHA, temp=TEMPERATURE):
    """Combined hard + soft loss for binary knowledge distillation.

    Args:
        y_true: tensor of shape (batch, 2) holding [hard_label, teacher_prob].
        y_pred: tensor of shape (batch, 1) with the student's sigmoid output.
        alpha:  relative weight of the soft (teacher) loss.
        temp:   distillation temperature.

    Returns:
        Scalar loss = (1 - alpha) * BCE(hard) + alpha * T^2 * KL(soft).
    """
    y_hard = y_true[:, 0]       # ground-truth label
    y_soft = y_true[:, 1]       # teacher sigmoid probability
    y_pred_hard = y_pred[:, 0]  # student sigmoid output

    # 1. Hard loss: standard binary cross-entropy against the true label.
    hard_loss = tf.keras.losses.binary_crossentropy(y_hard, y_pred_hard)

    # 2. Soft loss: KL divergence between temperature-softened distributions.
    eps = tf.keras.backend.epsilon()

    def to_logit(p):
        p = tf.clip_by_value(p, eps, 1.0 - eps)
        return tf.math.log(p / (1.0 - p))

    # Recover logits from probabilities, soften with temperature, then softmax.
    soft_targets = tf.nn.softmax(to_logit(y_soft) / temp)
    soft_predictions = tf.nn.softmax(to_logit(y_pred_hard) / temp)

    soft_loss = tf.keras.losses.KLDivergence()(soft_targets, soft_predictions)
    soft_loss = soft_loss * (temp ** 2)  # gradient-magnitude correction (Hinton 2015)

    # 3. Weighted combination.
    return (1.0 - alpha) * hard_loss + alpha * soft_loss


def build_student_model(n_features=10, hidden=(16, 8), dropout=0.1, lr=0.01):
    """Re-create the distilled student network (default: 16-8-1, ~290 params)."""
    layers = [tf.keras.layers.Dense(hidden[0], activation='tanh',
                                    input_shape=(n_features,)),
              tf.keras.layers.Dropout(dropout)]
    for units in hidden[1:]:
        layers += [tf.keras.layers.Dense(units, activation='tanh'),
                   tf.keras.layers.Dropout(dropout)]
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid', name='student_output'))

    model = tf.keras.models.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=distillation_loss, metrics=['accuracy'])
    return model

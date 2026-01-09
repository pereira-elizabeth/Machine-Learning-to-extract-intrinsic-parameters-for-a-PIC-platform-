import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from picml.utils import learning_rate_scheduler


from .models import (
    create_model_A_1,
    create_model_A_2,
    create_model_B_1,
    create_model_B_2,
    create_model_C_1,
    create_model_C_2,
)

# ======================
#   Case A
# ======================

def train_case_A_model1(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    epochs: int = 200,
    batch_size: int = 16,
    validation_split: float = 0.2,
):
    """
    Case A – model 1 (dense model on scaled spectra).

    Assumes:
        - X_train_scaled, X_test_scaled have shape (N, n_features)
        - y_train_scaled, y_test_scaled have matching first dimension.
    """
    model = create_model_A_1(input_shape=X_train_scaled.shape[1:])
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    test_loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    return model, history.history, test_loss


def train_case_A_model2(
    new_X_train_scaled_cnn: np.ndarray,
    new_y_train_scaled: np.ndarray,
    new_X_test_scaled_cnn: np.ndarray,
    new_y_test_scaled: np.ndarray,
    epochs: int = 100,
    batch_size: int = 40,
    validation_split: float = 0.2,
):
    """
    Case A – model 2 (CNN on theory-generated spectra).

    Expects:
        - new_X_*_scaled of shape (N, 40)  (or similar),
          will be expanded with an extra channel dimension for Conv1D.
        - new_y_*_scaled matching first dimension.
    """
    

    # If your model builder expects an input_shape like (40, 1)
    model_sc = create_model_A_2((40,1))
    model_sc.compile(optimizer="adam", loss="mse")

    history_sc = model_sc.fit(
        new_X_train_scaled_cnn,
        new_y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    test_loss = model_sc.evaluate(new_X_test_scaled_cnn, new_y_test_scaled, verbose=0)
    return model_sc, history_sc.history, test_loss


# ======================
#   Case B
# ======================

def train_case_B_model1(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    validation_split: float = 0.2,
    use_lr_schedule: bool = True,
):
    """
    Case B – model 1 (dense/MLP or similar on scaled spectra).

    Uses an optional learning-rate scheduler.
    """
    model = create_model_B_1(input_shape=X_train_scaled.shape[1:])
    model.compile(optimizer="adam", loss="mse")

    callbacks = []
    if use_lr_schedule:
        lr_scheduler_callback = LearningRateScheduler(learning_rate_scheduler)
        callbacks.append(lr_scheduler_callback)

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    return model, history.history, test_loss


def train_case_B_model2(
    new_X_train_scaled_cnn: np.ndarray,
    new_y_train_scaled: np.ndarray,
    new_X_test_scaled_cnn: np.ndarray,
    new_y_test_scaled: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    validation_split: float = 0.2,
):
    """
    Case B – model 2 (CNN on theory-generated spectra).

    Uses simple_spectrum_cnn() as the architecture.
    """


    model_sc = create_model_B_2()  # if you later change it to accept input_shape, adjust here
    model_sc.compile(optimizer="adam", loss="mse")

    history_sc = model_sc.fit(
        new_X_train_scaled_cnn,
        new_y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    test_loss = model_sc.evaluate(new_X_test_scaled_cnn, new_y_test_scaled, verbose=0)
    return model_sc, history_sc.history, test_loss


# ======================
#   Case C
# ======================

def train_case_C_model1(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    epochs: int = 200,
    batch_size: int = 16,
    validation_split: float = 0.2,
    use_lr_schedule: bool = True,
):
    """
    Case C – model 1 (multi-output regression on scaled spectra).

    y_*_scaled are expected to have shape (N, 2) for (phase, loss) or similar.
    """
    model = create_model_C_1(input_shape=X_train_scaled.shape[1:])
    model.compile(optimizer="adam", loss="mse")

    callbacks = []
    if use_lr_schedule:
        lr_scheduler_callback = LearningRateScheduler(learning_rate_scheduler)
        callbacks.append(lr_scheduler_callback)

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)

    # NOTE: inverse transforms and grouping are done in the notebook,
    # where target_scaler / target_scaler1 are available.
    return model, history.history, test_loss


def train_case_C_model2(
    new_X_train_scaled_cnn: np.ndarray,
    new_y_train_scaled: np.ndarray,
    new_X_test_scaled_cnn: np.ndarray,
    new_y_test_scaled: np.ndarray,
    epochs: int = 150,
    batch_size: int = 25,
    validation_split: float = 0.2,
):
    """
    Case C – model 2 (CNN on theory-generated spectra).

    Assumes new_X_*_scaled have shape (N, n_freqs), will be expanded
    with a singleton channel dimension.
    """
    model_sc = create_model_C_2(input_shape=new_X_train_scaled_cnn.shape[1:])
    model_sc.compile(optimizer="adam", loss="mse")

    history_sc = model_sc.fit(
        new_X_train_scaled_cnn,
        new_y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    test_loss = model_sc.evaluate(new_X_test_scaled_cnn, new_y_test_scaled, verbose=0)
    return model_sc, history_sc.history, test_loss

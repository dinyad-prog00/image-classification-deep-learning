import fidle
import tensorflow as tf

def get_callbacks(run_dir):
    fidle.utils.mkdir(run_dir + '/models')
    fidle.utils.mkdir(run_dir + '/logs')

    # ---- Callback tensorboard
    log_dir = run_dir + "/logs/tb_" + fidle.Chrono.tag_now()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # ---- Callback ModelCheckpoint - Save best model
    save_dir = run_dir + "/models/best-model.h5"
    bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir, verbose=0, monitor='accuracy', save_best_only=True)

    # ---- Callback ModelCheckpoint - Save model each epochs
    save_dir = run_dir + "/models/model-{epoch:04d}.h5"
    savemodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir, verbose=0)

    return [tensorboard_callback, bestmodel_callback, savemodel_callback]


def get_callbacks_exp(run_dir,tag_id,d_name,m_name,g_name):

    log_dir = f'{run_dir}/logs_{tag_id}/tb_{d_name}_{m_name}_{g_name}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # ---- Callbacks bestmodel
    save_dir = f'{run_dir}/models_{tag_id}/model_{d_name}_{m_name}_{g_name}.h5'
    bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir, verbose=0, monitor='accuracy', save_best_only=True)
   
    return [tensorboard_callback, bestmodel_callback]
    
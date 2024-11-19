import tensorflow as tf
import tf2onnx

# Load your Keras HDF5 model
model = tf.keras.models.load_model(
    '/home/patrick/Work/AutoQuake_Focal_pamicoding/DiTing-FOCALFLOW/models/DiTingMotionJul.hdf5',
    compile=False,
)

# Convert the model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model)

# Save the ONNX model to a file
with open(
    '/home/patrick/Work/AutoQuake_Focal_pamicoding/DiTing-FOCALFLOW/models/DiTingMotionJul.onnx',
    'wb',
) as f:
    f.write(onnx_model.SerializeToString())

print('Model successfully converted to ONNX format')
# %%
import onnx

model = onnx.load(
    '/home/patrick/Work/AutoQuake_Focal_pamicoding/DiTing-FOCALFLOW/models/DiTingMotionJul.onnx'
)

# Check that the model is well-formed
onnx.checker.check_model(model)

# Print a human-readable representation of the model
print(onnx.helper.printable_graph(model.graph))
# %%

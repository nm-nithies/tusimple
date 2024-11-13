import onnxruntime as ort
import numpy as np
ort_sess = ort.InferenceSession('ag_news_model.onnx')
outputs = ort_sess.run(None, {'input': text.numpy(),
                            'offsets':  torch.tensor([0]).numpy()})
# Print Result
result = outputs[0].argmax(axis=1)+1
print("This is a %s news" %ag_news_label[result[0]])

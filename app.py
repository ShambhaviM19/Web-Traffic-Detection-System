import gradio as gr
import numpy as np
import joblib 

def predict(region1, region2, region3, region4, region5, region6): 
    model = joblib.load('./localoutlierfactor.joblib')
    yhat = model.predict(np.array([[region1, region2, region3, region4, region5, region6]]))
    return 'Weird Traffic' if yhat == -1 else 'Normal Traffic'

app = gr.Interface(
    title='Traffic Anomaly Detection 🚦',
    fn=predict, 
    inputs=[gr.Slider(0,800),gr.Slider(0,800),gr.Slider(0,800),gr.Slider(0,800),gr.Slider(0,800),gr.Slider(0,800)], 
    outputs='text'
)

app.launch()
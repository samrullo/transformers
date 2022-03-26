text="""Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron 
instead! As a lifelong enemy of the Decepticons, I hope you can understand
my dilemma. To resolve the issue, I demand an exchange of Megatron for the Optimus Prime
figure I ordered. Enclosed are copies of my records concerning this purchase. I expect to hear
from you soon. Sincerely, Bumblebee."""

from transformers import pipeline
classifier=pipeline("text-classification")
import pandas as pd
outputs=classifier(text)
sentiments_df=pd.DataFrame(outputs)
print(sentiments_df)
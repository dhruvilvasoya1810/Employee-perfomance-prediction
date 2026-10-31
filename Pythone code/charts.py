import pandas as pd
import numpy as n
import dtale as d
import sweetviz as s
import matplotlib.pyplot as pt
import seaborn as sns
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


df=pd.read_csv("employe.csv")
df.info()

#Histogram

'''pt.figure()
ex=df.Experience
sns.histplot(ex,kde=True)
pt.show()'''

'''#Dtale 

d=d.show(df)
d.open_browser()    

#sweetwize data

r=s.analyze(df)
r.show_html()'''


#scatter chart

'''pt.scatter(df["Experience"],df["Salary"])
pt.show()

#Box plot

pt.figure(figsize=(15,8))
numeric_df=df.select_dtypes(include=['number'])
for i,col in enumerate(numeric_df.columns):
    pt.subplot(2,4,i+1)
    pt.boxplot(numeric_df[col])
    pt.title(col)
pt.tight_layout()
pt.show()'''


print("\nGenerating Sweetviz Report...")

report = s.analyze(df)

report.show_html(
    "SWEETVIZ_REPORT.html",
    open_browser=True
)

print("Sweetviz report generated successfully!")



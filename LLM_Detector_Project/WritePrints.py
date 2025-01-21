from writeprints_static import WriteprintsStatic
import pandas as pd

filepath = "datasets/{}_text.csv".format('wp')
output_filepath="{}_style.csv".format('wp')

df = pd.read_csv(filepath)

text_datas = list(df['text'])

vec = WriteprintsStatic()
X = vec.transform(text_datas).toarray()
print(vec.get_feature_names())

output_df = pd.DataFrame(data=X,columns=vec.get_feature_names())
output_df.to_csv(output_filepath,index=False,encoding='utf-8')

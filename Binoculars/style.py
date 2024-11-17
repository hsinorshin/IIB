from writeprints_static import WriteprintsStatic
import pandas as pd
import sys

filepath = "{}_text.csv".format(sys.argv[1])
output_filepath="{}_style.csv".format(sys.argv[1])

df = pd.read_csv(filepath)

text_datas = list(df['text'])

vec = WriteprintsStatic()
X = vec.transform(text_datas).toarray()

output_df = pd.DataFrame(data=X,columns=vec.get_feature_names())
output_df.to_csv(output_filepath,index=False,encoding='utf-8')

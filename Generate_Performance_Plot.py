"""
Generating charts for abstract
"""
import pandas as pd
import plotly.express as px

file_path = 'C:\\Users\\Sam\\Desktop\\GlomAnnotationsAndCode\\Outputs\\Student_Performance.csv'
file_df = pd.read_csv(file_path)
#print(file_df)

fig = px.bar(file_df,x='Including or Ignoring',y='Agreement Score', color = 'Student', barmode='group',text_auto=True,height=900)
fig.update_traces(textfont_size=10)
fig.write_image(file_path.replace('Student_Performance.csv','Results_figure_all.png'))

new_df = pd.concat([file_df[file_df['Including or Ignoring'].str.contains('Including')].groupby(by='Student').mean(),
                    file_df[file_df['Including or Ignoring'].str.contains('Ignoring')].groupby(by='Student').mean()],axis=0,ignore_index=False)
print(new_df)
new_df = new_df.reset_index()
new_df.index = ['Including Unscored']*3+['Ignoring Unscored']*3
print(new_df)
new_df = new_df.reset_index()
print(new_df)
new_df.columns = ['Unscored Cells Included?','Student #', 'Agreement with Expert']
new_df['Student #'] = new_df['Student #'].map({'Akshita':'Student 1','Jamie':'Student 2','Myles':'Student 3'})
print(new_df)
fig2 = px.bar(new_df, x='Student #',y='Agreement with Expert',color = 'Unscored Cells Included?',barmode='group',text_auto=True,title='Agreement Scores for each Student <br><sup>Including Unscored Cells and Ignoring</sup>')
fig2.write_image(file_path.replace('Student_Performance.csv','Results_figure_mean.png'))
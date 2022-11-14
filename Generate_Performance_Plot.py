"""
Generating charts for abstract
"""
import pandas as pd
import plotly.express as px

# Path to per-student performance values
file_path = 'C:\\Users\\Sam\\Desktop\\GlomAnnotationsAndCode\\Outputs\\Student_Performance.csv'
file_df = pd.read_csv(file_path)

# Making full-stacked group bar plot (stacked for each image)
fig = px.bar(file_df,x='Including or Ignoring',y='Agreement Score', color = 'Student', barmode='group',text_auto=True,height=900)
fig.update_traces(textfont_size=10)
fig.write_image(file_path.replace('Student_Performance.csv','Results_figure_all.png'))

# Grouping together scores for each student and separating by whether unscored cells were included in the score or not
new_df = pd.concat([file_df[file_df['Including or Ignoring'].str.contains('Including')].groupby(by='Student').mean(),
                    file_df[file_df['Including or Ignoring'].str.contains('Ignoring')].groupby(by='Student').mean()],axis=0,ignore_index=False)

# Messing around with the index, this could probably be simplified by just adding the columns the normal way
new_df = new_df.reset_index()
new_df.index = ['Including Unscored']*3+['Ignoring Unscored']*3
new_df = new_df.reset_index()
new_df.columns = ['Unscored Cells Included?','Student #', 'Agreement with Expert']

# Cool mapping thing to rename the student column according to a dictionary instead of replacing each one individually
new_df['Student #'] = new_df['Student #'].map({'Akshita':'Student 1','Jamie':'Student 2','Myles':'Student 3'})

# Making the mean agreement score bar plot separated by student and whether unscored cells were included or not
fig2 = px.bar(new_df, x='Student #',y='Agreement with Expert',color = 'Unscored Cells Included?',barmode='group',text_auto=True,title='Agreement Scores for each Student <br><sup>Including Unscored Cells and Ignoring</sup>')
fig2.write_image(file_path.replace('Student_Performance.csv','Results_figure_mean.png'))
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt

ground_truth = pd.read_csv("Ground Truth (Assignment 2).csv", skiprows=2, header=None)
# ground_truth.dropna(how="all", inplace=True)
result_data = pd.read_csv('resp1.csv', header=None, skiprows=1)
print(result_data.head())
y_true = ground_truth.iloc[0:1495,11]
y_pred = result_data.iloc[0:1495,0]
# y_pred = np.array(:0)
# print(list(y_true))
# print(y_true)
# print(y_pred)
q1_f1score = precision_recall_fscore_support(y_true, y_pred, average='macro')
print(q1_f1score)

ground_truth_q2 = ground_truth.loc[ground_truth[11].isin([1,2])]
frames = ground_truth_q2[0].tolist()
result_data_q2 = result_data.loc[result_data[1].isin(frames)]
print(ground_truth_q2.shape, result_data_q2.shape)
q2_f1score = precision_recall_fscore_support(result_data_q2[9].tolist()+result_data_q2[10].tolist(), ground_truth_q2[12].tolist()+ground_truth_q2[13].tolist(), average='macro')
print(q2_f1score)

color = precision_recall_fscore_support(result_data_q2[24].tolist()
                                        +result_data_q2[25].tolist() +
                                        result_data_q2[26].tolist() + 
                                        result_data_q2[27].tolist() +
                                        result_data_q2[28].tolist() 
                                        ,ground_truth_q2[14].tolist()
                                        +ground_truth_q2[15].tolist() +
                                        ground_truth_q2[16].tolist() + 
                                        ground_truth_q2[17].tolist() +
                                        ground_truth_q2[18].tolist() 
, average='macro')
print(color)
# labels = ["Q1","Q2","Q3"]
# fig, ax = plt.subplots()
r1 = np.arange(3)
barWidth = 0.25
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]


plt.bar(r1, [q1_f1score[0],q2_f1score[0],color[0]], color = ["#E16446"], width=barWidth, edgecolor='white', label='Precision')
plt.bar(r2, [q1_f1score[1],q2_f1score[1],color[1]], color = ["#39C638"], width=barWidth, edgecolor='white', label='Recall')
plt.bar(r3, [q1_f1score[2],q2_f1score[2],color[2]], color = ["#2372A9"], width=barWidth, edgecolor='white', label='F1-Score')
# plt.xlabel("Query")
# plt.ylabel("F1 Score")
plt.title('Event Query Accuracy')
plt.xticks([r + barWidth for r in range(3)], ['Stage 1', 'Stage 2', 'Stage 3'])
plt.legend()
plt.show()
plt.title('Event Extraction Time for Q1')
plt.plot(result_data[21].tolist())
plt.show()
plt.title('Event Extraction Time for Q2')
plt.plot(result_data[22].tolist())
plt.show()
plt.title('Event Extraction Time for Q3')
plt.plot(result_data[23].tolist())
plt.show()


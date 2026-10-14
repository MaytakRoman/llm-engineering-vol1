TP, FP, FN, TN = 90, 20, 10, 880
precision = TP/(TP+FP)
recall = TP/(TP+FN)
fpr = FP/(FP+TN)
assert TP+FP+FN+TN == 1000
print("precision, recall, fpr:", precision, recall, fpr)
for prevalence in (.1, .01):
    ppv = recall*prevalence
    ppv /= recall*prevalence + fpr*(1-prevalence)
    print("prevalence", prevalence, "ppv", ppv)

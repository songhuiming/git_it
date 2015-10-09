def roc_auc(x, y):
	unq_x = np.unique(x)
	n1 = sum(y)
	n = len(y)
	Sens = np.zeros_like(unq_x)
	Spec = np.zeros_like(unq_x)
	for j, u in enumerate(unq_x):
		Sens[j] = np.sum((x >= u) * y) / float(n1)
		Spec[j] = np.sum((x <= u) *(1 - y)) / float(n - n1)
	auc = 0.0
	for i in range(len(Spec) - 1):
		auc += (Spec[i + 1] - Spec[i]) * (Sens[i + 1] + Sens[i]) / 2.0
	plt.plot(1-Spec, Sens, '-')
	plt.plot([0, 1], [0, 1], '-', color = 'grey')
	plt.xlabel("1 - Specificity", size = 17)
	plt.ylabel("Sensitivity", size = 17)
	plt.title("AUC = %.2f" %auc)
	plt.show()
	print "AUC is %s" %(auc)

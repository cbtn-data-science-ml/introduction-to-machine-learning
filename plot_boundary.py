# Plot Boundary
def plot_boundary(X, y, model):
  x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5 # min/max values for Feature_1
  y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5 # min/max values for Feature_2
  xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01), torch.arange(y_min, y_max, 0.01)) # Create mesh grid

  # Predict Over Mesh Grid
  grid = torch.column_stack((xx.ravel(), yy.ravel())).float()
  with torch.no_grad(): # again no need to calculate gradients here as well
    preds = torch.sigmoid(model(grid)).reshape(xx.shape) # makes predictions


    # Plot Contour and Training Samples
    plt.contourf(xx, yy, preds, alpha=0.7, cmap=plt.cm.RdBu, levels=np.linspace(0, 1, 50)) # Decision boundary
    plt.colorbar() # Add Probability Colorbar
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdBu, edgecolors="k") # places actual datapoints on top of boundary
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary with Probabilty")
    plt.show()

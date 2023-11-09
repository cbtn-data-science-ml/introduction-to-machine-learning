def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    spacing = min(x_max - x_min, y_max - y_min) / 100

    # Generate a grid of points with distance 'spacing' between them
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))

    # Flatten the grid to pass into model
    data = np.hstack((XX.ravel().reshape(-1, 1),
                      YY.ravel().reshape(-1, 1)))

    # Convert to tensor
    data_tensor = torch.FloatTensor(data).to(device)

    # Predict the function value for the whole grid
    model.eval()
    with torch.no_grad():
        Z = model(data_tensor)
        Z = Z.reshape(XX.shape)
        Z = torch.sigmoid(Z).cpu().numpy()

    plt.contourf(XX, YY, Z, levels=[0, 0.5, 1], cmap='RdBu', alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, lw=0)
    plt.xlim(XX.min(), XX.max())
    plt.ylim(YY.min(), YY.max())

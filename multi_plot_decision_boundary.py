from matplotlib.lines import Line2D  # Add this line
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    spacing = min(x_max - x_min, y_max - y_min) / 100

    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing), np.arange(y_min, y_max, spacing))
    data = np.hstack((XX.ravel().reshape(-1, 1), YY.ravel().reshape(-1, 1)))
    data_tensor = torch.FloatTensor(data).to(device)

    model.eval()
    with torch.no_grad():
        Z = model(data_tensor)
        Z = Z.argmax(dim=1).cpu().numpy()

    # Define class labels and colors
    class_labels = ["Mammal", "Bird", "Reptile"]
    class_colors = ['blue', 'green', 'red']

    # Create a legend with class labels and colors
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
                       for label, color in zip(class_labels, class_colors)]

    plt.contourf(XX, YY, Z.reshape(XX.shape), levels=np.arange(Z.max() + 2) - 0.5, cmap='RdBu', alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, lw=0)
    plt.xlim(XX.min(), XX.max())
    plt.ylim(YY.min(), YY.max())

    # Set the background color to green
    plt.gca().set_facecolor('green')

    # Add the legend to the top left corner
    plt.legend(handles=legend_elements, loc='upper left')

    plt.show()

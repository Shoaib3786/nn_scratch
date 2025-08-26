
class Accuracy:
  def calculate(self, y_pred, y_actual):
    pred_classes = (y_pred > 0.5).float().view(-1)   # flatten to (batch,)
    y_actual = y_actual.view(-1).float()             # also flatten
    return (pred_classes == y_actual).float().mean()
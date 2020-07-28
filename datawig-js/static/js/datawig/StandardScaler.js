
class StandardScaler {
  // X_: tf.tensor2d
  fit_transform(X_) {
    const n = X_.shape[0]
    const d = X_.shape[1]

    const u = tf.mean(X_, 0).tile([n]).reshape([n, d])
    const X_less_u = X_.sub(u)
    const std = tf.sqrt(tf.moments(X_, 0).variance)
    const normalized = X_less_u.div(std.tile([n]).reshape([n, d]))
    return normalized
  }
}

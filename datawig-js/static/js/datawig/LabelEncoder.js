
/**
 * Implements a standard one-hot label encoder.
 */
class LabelEncoder {
  /**
   * Construct a new label encoder for a specific column.
   * @param {String}  column  Name of the column to encode
   */
  constructor(column) {
    this.column = column
    this.label_indices = new Object()
    this.label_counter = 0
    this.index_labels = new Object()
  }

  is_fitted() {
    return this.label_counter > 0
  }

  /**
   * Fit to the data.
   * @param   {Array[Object]}  dataset  Array of Objects, each of the objects must
   *                                    contain this.column property
   * @return  {void}
   */
  fit(dataset) {
    const t0 = performance.now()

    const n = dataset.length
    console.log('Fitting to ', n, 'rows')
    for(var i = 0; i < n; i++) {
      const label = dataset[i][this.column]
      if (! this.label_indices.hasOwnProperty(label)) {
        this.label_indices[label] = this.label_counter
        this.index_labels[this.label_counter] = label
        this.label_counter++
      }
    }

    const t1 = performance.now()
    console.log("LabelEncoder.fit took " + (t1 - t0) + " milliseconds.")
  }

  /**
   * Transform a given dataset.
   * @param   {Array[Object]}  dataset  Array of Objects, each of the objects must
   *                                    contain this.column property
   * @return  {tf.tensor2d}             2 dimensional (n by k) tensorflow-js
   *                                    tensor with n as number of elements in
   *                                    the dataset and k the number of distinct
   *                                    labels that are known to the encoder
   */
  transform(dataset) {
    const t0 = performance.now()

    const n = dataset.length
    const d = this.label_counter
    const encoded = new Array(n * d).fill(0)

    for(var i = 0; i < n; i++) {
      const label = dataset[i][this.column]
      if (this.label_indices.hasOwnProperty(label)) {
        const index = this.label_indices[label]
        encoded[(i*d)+index] = 1
      }
    }

    const t1 = performance.now()
    console.log("LabelEncoder.transform took " + (t1 - t0) + " milliseconds.")

    return tf.tensor2d(encoded, [n, d])
  }

  fit_transform(dataset) {
    this.fit(dataset)
    return this.transform(dataset)
  }

}

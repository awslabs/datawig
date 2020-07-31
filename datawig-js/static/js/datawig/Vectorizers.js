class BinaryVectorizer {
  constructor(column) {
    this.column = column
    // holds all distinct tokens
    this.tokens = new Set()
    this.token_indices = new Object()
    this.index_tokens = new Object()
  }

  is_fitted() {
    return this.tokens.size > 0
  }

  fit(documents) {
    // console.log(documents)
    for (var i = 0; i < documents.length; i++) {
      // console.log(documents[i])
      const tokens_ = documents[i][this.column].toLowerCase().split(' ')
      for (var j = 0; j < tokens_.length; j++) {
        // console.log(tokens_, tokens_[j])
        if (tokens_[j] != '') {
          this.tokens.add(tokens_[j])
        }
      }
    }

    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set
    let token_index = 0
    for (const item of this.tokens.values()) {
      this.token_indices[item] = token_index
      this.index_tokens[token_index] = item
      token_index++
    }
  }

  // array[obj]
  fit_transform(documents) {
    const t0 = performance.now()

    this.fit(documents)

    const t1 = performance.now()
    console.log("BinaryVectorizer.fit took " + (t1 - t0) + " milliseconds.")

    return this.transform(documents)
  }

  // array[obj]
  transform(documents) {
    const t0 = performance.now()

    const d = this.tokens.size
    // https://stackoverflow.com/questions/34937349/javascript-create-empty-array-of-a-given-size/41246860
    const encodings = Array(documents.length * d).fill(0)

    for (var i = 0; i < documents.length; i++) {
      const tokens_ = documents[i][this.column].toLowerCase().split(' ')
      for (var j = 0; j < tokens_.length; j++) {
        encodings[(i * d) + this.token_indices[tokens_[j]]] = 1
      }
    }

    const t1 = performance.now()
    console.log("BinaryVectorizer.transform took " + (t1 - t0) + " milliseconds.")

    return tf.tensor2d(encodings, [documents.length, this.tokens.size])
  }

  // X: tf.tensor2d
  inverse_transform(X) {
    const n = X.shape[0]
    const d = X.shape[1]
    const document_tokens = new Array(n)
    const X_ = X.dataSync()

    for (let i = 0; i < n; i++) {
      let i_tokens = new Array()
      for (let j = 0; j < d; j++) {
        const x = X_[(i*d) + j]
        if (x != 0) {
          i_tokens.push(this.index_tokens[j])
        }
      }

      document_tokens[i] = i_tokens

    }
    return document_tokens
  }
}

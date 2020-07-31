class Math_ {
  /**
   * Returns sorted indices of input
   *
   * https://stackoverflow.com/questions/46622486/what-is-the-javascript-equivalent-of-numpy-argsort
   * @param {Array}
   * @return {Array}
   */
  static argsort(array, ascending = true) {
    let order
    if (ascending) {
      order = 1
    } else {
      order = -1
    }

    return array
      .map(function(value, index) { return [value, index] })
      .sort(function(vi1, vi2) { return order * (vi1[0] - vi2[0]) })
      .map(function(vi) { return vi[1] })
  }

  /**
   * Shuffles array in place.
   * https://stackoverflow.com/questions/6274339/how-can-i-shuffle-an-array
   *
   * @param {Array} a items An array containing the items.
   */
  static shuffle(a) {
      var j, x, i;
      for (i = a.length - 1; i > 0; i--) {
          j = Math.floor(Math.random() * (i + 1));
          x = a[i];
          a[i] = a[j];
          a[j] = x;
      }
      return a;
  }
}

function hasLabel(object, attribute) {
  // https://stackoverflow.com/questions/135448/how-do-i-check-if-an-object-has-a-specific-property-in-javascript
  return object.hasOwnProperty(attribute) && object[attribute] != '' && object[attribute] != undefined
}

function showSampleTable(els) {
  const columns = Object.getOwnPropertyNames(els[0]).map(function(el) {
    return {field: el, title: el, sortable: true}
  })

  const data = els.slice(0, 1000).map(function(el) {
    var entry = new Object()
    columns.forEach(function(col) {
      entry[col.field] = el[col.field]
    })
    return entry
  })

  $('#table').bootstrapTable({
    search: true,
    pagination: true,
    columns: columns,
    data: data
  })
}

function explainInput(explanations) {
  // Read the keyword
  const query = $('#query').val()

  const options = {
    separateWordSearch: true,
    diacritics: true,
    accuracy: 'exactly'
  }

  $('#explanation').text(query)

  const to_mark = explanations.map(function(el) { return el['token'] }).join(' ')

  // Remove previous marked elements and mark
  // the new keyword inside the context
  $("#explanation").unmark({
    done: function() {
      $("#explanation").mark(to_mark, options)
    }
  })
}

function readFile(file){
  return new Promise((resolve, reject) => {
    var fr = new FileReader();
    fr.onload = () => {
      resolve(fr.result)
    };
    fr.readAsBinaryString(file);
  });
}

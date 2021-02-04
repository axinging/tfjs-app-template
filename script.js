import settings from 'config.json';


function cartesianProduct(arr) {
  return arr.reduce(function(a,b){
      return a.map(function(x){
          return b.map(function(y){
              return x.concat([y]);
          })
      }).reduce(function(a,b){ return a.concat(b) },[])
  }, [[]])
}

console.log(settings.workloads);

var a = cartesianProduct([[1, 2,3], [4, 5,6], [7, 8], [9,10]]);
console.log(JSON.stringify(a));
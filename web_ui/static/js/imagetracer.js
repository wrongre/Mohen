/* Placeholder imagetracer.js
   Minimal stub to avoid 404 requests when SVG export is not used.
   Provides a tiny ImageTracer.trace API that returns an empty SVG.
*/
(function(global){
  function noopTrace(image, options, callback){
    var result = { svg: '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>' };
    if(typeof callback === 'function') callback(result);
    return result;
  }

  // Export a minimal ImageTracer interface expected by the UI
  global.ImageTracer = global.ImageTracer || {};
  global.ImageTracer.trace = noopTrace;
  // keep compatibility for other possible helpers
  global.ImageTracer.getSvgString = function(){ return '<svg xmlns="http://www.w3.org/2000/svg"/>'; };
})(this);

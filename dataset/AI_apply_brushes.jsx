// apply multiple brushes to the sketch and export then to png images
// set up export path
var pathStrSplit = $.fileName.split('/');
pathStrSplit.pop(); // pop out the current file name, we don't need it
pathStrSplit.pop();

var currPath = pathStrSplit.join('/');
var svgPath = currPath + "/dataset/sample/svg/sketch/";
var aiPath = currPath + "/dataset/sample/ai/";
var pngPath = currPath + "/dataset/sample/train/";

// set up the export option
var options = new ImageCaptureOptions();
options.artBoardClipping = true;
options.resolution = 300; // dpi 300
options.antiAliasing = true;
options.matte = false;
options.horizontalScale = 100;
options.verticalScale = 100;
options.transparency = true; // I hope this could be helpful when adding paper textures

// apply brushes
for (var i = 0; i < 9999999; i++){
    // apply one brush, but what brushes should we apply?
    for (var k = 0; k< 10; k++){
        var aiInput = aiPath + ("0000000" + i).slice(-7) + ".ai";
        var pngInput = pngPath + ("00" + k).slice(-2) + "/" + ("0000000" + i).slice(-7) + ".png";
        var ai = new File(aiInput);
        var png = new File(pngInput);
        if (png.exists) continue;
        if (ai.exists && !png.exists){
            var docAI = app.open(ai);     
            if (app.activeDocument.name != docAI.name) docAI.activate();
            var activeAB = docAI.artboards[docAI.artboards.getActiveArtboardIndex()];
            // apply one brush
            docAI.selectObjectsOnActiveArtboard();
            var brush1 = app.activeDocument.brushes[0];
            var j = 0;
            for (; j < app.activeDocument.pageItems.length; j++){
                brush1.applyTo(app.activeDocument.pageItems[j]);
            }
            docAI.imageCapture(png, activeAB.artboardRect, options);
        }
        docAI.close(SaveOptions.DONOTSAVECHANGES);
    }   
}
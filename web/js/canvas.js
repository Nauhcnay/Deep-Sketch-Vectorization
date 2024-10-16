// define different layer index
const LAYER_RASTER = 0
const LAYER_VEC_RAW = 1
const LAYER_VEC_REFINE = 2
const LAYER_VEC_FINAL = 3
const LAYER_KEYPOINT = 4
const LAYER_UNCERTAIN_USM = 5
const LAYER_USM = 6 
const LAYER_GRID = 7

// keypoint colors
const COLOR_END = 'green'
const COLOR_SHARP = 'red'
const COLOR_JUNC = 'blue'

const COLOR_CUBE = '#b14a7d'
const COLOR_USM_R = 177
const COLOR_USM_G = 74
const COLOR_USM_B = 125
const COLOR_USM_A = 178
const COLOR_UN_USM_R = 0 
const COLOR_UN_USM_G = 274
const COLOR_UN_USM_B = 0
const COLOR_UN_USM_A = 178

// opacity
OPACITY_KP = 0.5
const OPACITY_USM = 0.7

const INIT_RADIUS = 10.0  
CIRCLE_RADIUS = 10    
delta = 0;  // save zoom parameter

let _config = {
    canvasState             : [],
    managerState            : [],
    usmState                : [],
    currentStateIndex       : -1,
    undoStatus              : false,
    redoStatus              : false,
    undoFinishedStatus      : 1,
    redoFinishedStatus      : 1
};

function getZoom(){
    var max = 200;
    let zoom = window.__canvas.getZoom();
    if(delta != 0){
        zoom *= 0.999 ** delta;
        if (zoom > max) zoom = max;
        if (zoom < 0.01) zoom = 0.01;
    }
    return zoom;
}

function getScale(){
    var fixed = document.getElementById('c_kp_radius').checked;
    let scale = CIRCLE_RADIUS/INIT_RADIUS
    if(fixed){
        scale = CIRCLE_RADIUS/INIT_RADIUS*1/(2*getZoom());
    }
    return scale;
}

function addCircle(x, y, color, scale){
    // add keypoint as a circle
    var circle = new fabric.Circle({ 
                        radius: INIT_RADIUS, 
                        fill: color,
                        left: x,
                        top: y,
                        originY: 'center', 
                        originX: 'center',
                        scaleX: scale,
                        scaleY: scale,
                        opacity: OPACITY_KP,
                        strokeWidth: 0,
                        hasControls: false,
                        selectable: true
                        // hasBorders: false
                    });  
    window.__canvas.add(circle);
}

function updateUsmPixel(x, y, alwayDraw) {
    // if not alwayDraw, then switch on/off
    var pos = (y * raw_width + x) * 4;
    if(alwayDraw || buffer[pos] == 0){
        buffer[pos] = COLOR_USM_R;           
        buffer[pos+1] = COLOR_USM_G;           
        buffer[pos+2] = COLOR_USM_B;           
        buffer[pos+3] = COLOR_USM_A;
        trackUsmPixel(x, y, true);
    }
    else{
        buffer[pos] = 0;             
        buffer[pos+1] = 0;             
        buffer[pos+2] = 0;              
        buffer[pos+3] = 0; 
        trackUsmPixel(x, y, false);
    }
    idata.data.set(buffer);
    usmCtx.putImageData(idata, 0, 0, x, y, 1, 1);
}

function updateUncertainUsmPixel(x, y) {
    let index = usm_pixel_dict["r"+x+"c"+y];
    if(index !== undefined){
        updateUncertainUsmRegion(index);
    }
}

function updateUncertainUsmRegion(index) {
    // click a region number
    let region = usm_region_dict[index]['pos'];   
    let left=raw_width, right=0, top=raw_height, bottom=0; //get bounding box
    for(var i=0; i<region[0].length; i++){
        left = left>region[0][i]?region[0][i]:left;
        right = right<region[0][i]?region[0][i]:right;
        top = top>region[1][i]?region[1][i]:top;
        bottom = bottom<region[1][i]?region[1][i]:bottom;
        //
        var pos = (region[1][i] * raw_width + region[0][i]) * 4;
        buffer[pos] = COLOR_USM_R;           
        buffer[pos+1] = COLOR_USM_G;           
        buffer[pos+2] = COLOR_USM_B;           
        buffer[pos+3] = COLOR_USM_A;
        //
        trackUsmPixel(region[0][i], region[1][i], true);
    }
    idata.data.set(buffer);
    usmCtx.putImageData(idata, 0, 0, left, top, right-left+1, bottom-top+1);
}

function trackUsmPixel(x, y, value){
    let key = "r"+x+"c"+y;
    if(!usm_dirty_dict.hasOwnProperty(key)){
        usm_dirty_row.push(x);
        usm_dirty_column.push(y);
    }
    usm_dirty_dict[key] = value;
}

function displayLayer(index, hidden){
    var layer = window.manager.getLayer(index);
    var objs = window.__canvas.getObjects();
    var canvas = window.__canvas; 
    for (var i = layer.startIndex; i < layer.endIndex; i++) {
        if(hidden){
            objs[i].visible = false;
        }
        else{
            objs[i].visible = true;
        }
    } 
    canvas.renderAll();
}

function updateActiveLayer(){
    // priority: keypoint < usm < uncertain usm
    var checkUnusm = document.getElementById('c_uncertain_usm');
    var checkUsm = document.getElementById('c_usm');    
    // if (checkUnusm.checked == true)
    // {
    //     window.manager.activeLayer = window.manager.getLayer(LAYER_UNCERTAIN_USM);
    // }
    // else{
        if (checkUsm.checked == true)
        {
            window.manager.activeLayer = window.manager.getLayer(LAYER_USM);
        }
        else{
            window.manager.activeLayer = window.manager.getLayer(LAYER_KEYPOINT);
        } 
    // } 
}

function updateLayerEvt(index, enable){
    var layer = window.manager.getLayer(index);
    var objs = window.__canvas.getObjects();
    var canvas = window.__canvas; 
    for (var i = layer.startIndex; i < layer.endIndex; i++) {
        objs[i].evented = enable;
        if(index < LAYER_USM){
            // in case usm layer can be selectable
            objs[i].selectable = enable;
        }
    } 
    canvas.renderAll();
}

function initCanvas(){
    var canvas = this.__canvas = new fabric.Canvas('c', {
        isDrawingMode: false,
        imageSmoothingEnabled: false,
        backgroundColor: "#FFFFFF",
        perPixelTargetFind :true
    });
    // Create layer manager
    var manager = this.manager = new fabricLayer.LayerManager(canvas);
    // Add layers
    for (var i = 0; i < 7; i++) {
        manager.addLayer();
    } 
    fabric.Group.prototype.hasControls = false;  // disable muti-selection for keypoints
    fabric.Object.prototype.transparentCorners = false;
    // this setting is important, otherwise the svg result is blurry
    fabric.Object.prototype.objectCaching = false;  
    fabric.Object.prototype.noScaleCache = false;  
    // moving object
    canvas.on({
        'object:moving': function(e) {
        },
        'object:modified': function(e) {
        }
    });
    canvas.on('mouse:down', function(opt) {
        if(!canvas.isDrawingMode){
            var evt = opt.e;
            this.isDragging = true;
            var checkbox = document.getElementById('c_keypoints');
            var checkUsm= document.getElementById('c_usm');
            var checkUnusm= document.getElementById('c_uncertain_usm');
            if(checkbox.checked && window.manager.activeLayerIndex == LAYER_KEYPOINT){
                if(opt.target){
                    // click current keypoint
                }
                else{
                    log("create new keypoint: end point");
                    var pointer = window.__canvas.getPointer(opt.e);
                    // add new keypoint, default: end point
                    addCircle(pointer.x, pointer.y, COLOR_END, getScale());
                    window.__canvas.renderAll();
                    updateCanvasState();  // add keypoint
                }
            }
            else if(checkUsm && window.manager.activeLayerIndex == LAYER_USM){
                // if(opt.target){
                //     // click current svg path
                // }
                // else{
                    log("start pencil drawing");
                    var pointer = window.__canvas.getPointer(opt.e);
                    updateUsmPixel(Math.floor(pointer.x), Math.floor(pointer.y), false);
                    updateCanvasState();
                    window.__canvas.renderAll();
                // }
            }
            else if(checkUnusm && window.manager.activeLayerIndex == LAYER_UNCERTAIN_USM){
                if(opt.target){
                }
                else{
                    var pointer = window.__canvas.getPointer(opt.e);
                    updateUncertainUsmPixel(Math.floor(pointer.x), Math.floor(pointer.y));
                    window.__canvas.renderAll();
                }
            }
        }
    });
    canvas.on('mouse:move', function(opt) {
    });
    canvas.on('mouse:dblclick', function(opt) {
        if(!canvas.isDrawingMode){
            if(opt.target){
                if(window.manager.activeLayerIndex == LAYER_KEYPOINT){
                    if(opt.target.fill == COLOR_SHARP){
                        opt.target.fill = COLOR_JUNC;
                        log("change keypoint type: sharp turn -> junction");
                    }
                    else if(opt.target.fill == COLOR_END){
                        opt.target.fill = COLOR_SHARP;
                        log("change keypoint type: end point -> sharp turn");
                    }
                    else if(opt.target.fill == COLOR_JUNC){
                        opt.target.fill = COLOR_END;
                        log("change keypoint type: junction -> end point");
                    }
                    canvas.renderAll();
                    updateCanvasState();  // record
                }
            }
        }
    });
    canvas.on('mouse:up', function(opt) {
        this.isDragging = false;
    });
    // Zoom 
    canvas.on('mouse:wheel', function(opt){
        var max = 200;
        delta = opt.e.deltaY;
        var zoom = canvas.getZoom();
        zoom *= 0.999 ** delta;
        if (zoom > max) zoom = max;
        if (zoom < 0.01) zoom = 0.01;
        canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
        updateKpSize(zoom);  // update size
        opt.e.preventDefault();
        opt.e.stopPropagation();
    });
    // keypoint
    document.querySelector('#keypoint-radius').oninput = function(auto){
        if(auto !== true){
            log("change keypoint radius: " + this.value);
        }
        CIRCLE_RADIUS = parseInt(this.value, 10) || 10;
        this.previousSibling.innerHTML = this.value;
        var zoom = getZoom();
        updateKpSize(zoom);  // update size
    };
    document.querySelector('#keypoint-radius').oninput(true);
    // keypoint
    document.querySelector('#keypoint-opacity').oninput = function(auto){
        if(auto !== true){
            log("change keypoint opacity: " + this.value);
        }
        OPACITY_KP = parseFloat(this.value);
        this.previousSibling.innerHTML = this.value;
        updateKpOpacity();  
    };
    document.querySelector('#keypoint-opacity').oninput(true);
    // usm
    var drawingModeEl = $('#drawing-mode'),
    drawingOptionsEl = $('#drawing-mode-options'),
    drawingColorEl = $('#drawing-color'),
    drawingLineWidthEl = $('#drawing-line-width'),
    clearEl = $('#clear-canvas');
    clearEl.click(function() { canvas.clear() });

    drawingModeEl.click(function() {
        log("click lasso button");
        lassoBtnCnt++;
        document.getElementById("lasso-img").src="./icon/lasso.png";
        document.getElementById("pen-img").src="./icon/pen_blue.png";
        canvas.isDrawingMode = true;
        drawingOptionsEl.show();
    }); 

    $('#pen-mode').click(function() {
        log("click pencil button");
        penBtnCnt++;
        document.getElementById("pen-img").src="./icon/pen.png";
        document.getElementById("lasso-img").src="./icon/lasso_blue.png";
        canvas.isDrawingMode = false;
        drawingOptionsEl.hide();
    }); 

    var brush = new fabric.PencilBrush(canvas);
    brush.color = 'black';
    brush.width = 0.1;
    // brush.limitedToCanvasSize = true;
    canvas.freeDrawingBrush = brush;
    // canvas.freeDrawingBrush._finalizeAndAddPath();
    drawingColorEl.change(function() {
        var brush = canvas.freeDrawingBrush;
        brush.color = this.value;
        if (brush.getPatternSrc) {
            brush.source = brush.getPatternSrc.call(brush);
        }
    });
    drawingColorEl.change();
    drawingLineWidthEl.change(function() {
        canvas.freeDrawingBrush.width = parseInt(this.value, 10) || 0.1;
        this.previousSibling.innerHTML = this.value;
    });
    drawingLineWidthEl.change();

    canvas.on('path:created', function(options) {
        // 
        if(window.manager.activeLayerIndex == LAYER_USM){
            log("finish lasso drawing");
            var clean = document.getElementById('c_clean_usm').checked;
            var path = options.path;
            path.objectCaching = false;
            path.perPixelTargetFind = true;   // important: pixel rather than bounding box
            path.set('fill', 'rgba(0,255,0,0.3)');
            let rect = path.getBoundingRect(true, true);
            let left = Math.floor(rect.left);
            let top = Math.floor(rect.top);
            let w = Math.floor(rect.left+rect.width)-left+1;
            let h = Math.floor(rect.top+rect.height)-top+1;
            let cnt = 0;
            for (var i=0; i<w; i++) {
                for (var j=0; j<h; j++) {
                    let x = left+i;
                    let y = top+j;
                    let point = new fabric.Point(x+0.5, y+0.5); // position
                    let coord = fabric.util.transformPoint(point, canvas.viewportTransform); // position -> coordinate
                    if(path.containsPoint(point, null, true, true) && !canvas.isTargetTransparent(path, coord.x, coord.y)){
                        // draw pixel
                        var pos = (y * raw_width + x) * 4;
                        if(clean){
                            buffer[pos] = 0;           
                            buffer[pos+1] = 0;           
                            buffer[pos+2] = 0;           
                            buffer[pos+3] = 0;
                            trackUsmPixel(x, y, false);
                        }
                        else{
                            buffer[pos] = COLOR_USM_R;           
                            buffer[pos+1] = COLOR_USM_G;           
                            buffer[pos+2] = COLOR_USM_B;           
                            buffer[pos+3] = COLOR_USM_A;
                            trackUsmPixel(x, y, true);
                        }
                        cnt++;
                    }
                }
            }
            idata.data.set(buffer);
            usmCtx.putImageData(idata, 0, 0, left, top, w, h);
            // canvas.isDrawingMode = false;
            canvas.remove(path);
            updateCanvasState();
            log("finish lasso drawing, number of pixels updated: " + cnt);
        }
    });

}

function updateKpSize(zoom){
    var fixed = document.getElementById('c_kp_radius').checked;
    var layer = window.manager.getLayer(LAYER_KEYPOINT);
    var objs = window.__canvas.getObjects();
    var canvas = window.__canvas; 
    let newRadius = CIRCLE_RADIUS/(2*zoom);
    canvas.renderOnAddRemove = false;
    for (var i = layer.startIndex; i < layer.endIndex; i++) {
        // objs[i].radius = newRadius;
        if(fixed){
            objs[i].scaleX = objs[i].scaleY = (CIRCLE_RADIUS/INIT_RADIUS)*1/(2*zoom);
        }
        else{
            objs[i].scaleX = objs[i].scaleY = (CIRCLE_RADIUS/INIT_RADIUS)*1;
        }
    } 
    canvas.renderAll();
    canvas.renderOnAddRemove = true;
}

function updateKpOpacity(){
    var layer = window.manager.getLayer(LAYER_KEYPOINT);
    var objs = window.__canvas.getObjects();
    var canvas = window.__canvas; 
    for (var i = layer.startIndex; i < layer.endIndex; i++) {
        objs[i].opacity = OPACITY_KP;
    } 
    canvas.renderAll();
}

function recenter(){
    log("click recenter button");
    window.__canvas.setViewportTransform([ratio,0,0, ratio,0,0]); 
    updateKpSize(ratio);
}

function addGrids(){
    window.manager.activeLayer = window.manager.getLayer(LAYER_GRID);
    var canvas = window.__canvas;
    options = {
        distance: 1,
        width: raw_width,
        height: raw_height,
        param: {
            stroke: '#ebebeb',
            strokeWidth: 0.01,
            objectCaching: false,
            selectable: false,
            evented: false
        }
    };
    for (var i = 0; i <= options.width; i++) {
        var distance = i * options.distance;
        var vertical = new fabric.Line([ distance, 0, distance, options.height], options.param);
        vertical.set({stroke: '#cccccc'});
        canvas.add(vertical);
    };
    for (var j = 0; j <= options.height; j++) {
        var distance = j * options.distance;
        var horizontal = new fabric.Line([ 0, distance, options.width, distance], options.param);
        horizontal.set({stroke: '#cccccc'});
        canvas.add(horizontal);
    };
    canvas.renderAll();
}


function deleteObj(){
    log("click delete button");
    deleteBtnCnt++;
    var canvas = window.__canvas;
    var selectedObject = canvas.getActiveObject();
    window.__canvas.discardActiveObject();
    if (selectedObject) {
        if (Array.isArray(selectedObject._objects)){
            for (const element of selectedObject._objects) {
                canvas.remove(element);
            }
        }
        else{
            canvas.remove(selectedObject);
        }
        canvas.renderAll();
    }
    updateCanvasState();   // delete keypoint
};

function initHistory(){
    var canvas = window.__canvas;
    canvas.on(
        'object:modified', function(){
            updateCanvasState();
        }
    );
    canvas.on(
        'object:added', function(){
        }
    );
    canvas.on(
        'object:removed', function(){
        }
    );
    updateCanvasState();  // save original status
}

function createRegionImg(){
    document.querySelector('#unusm-div').style.display = "inline-block";
    let newCanvas = document.createElement('canvas');
    newCanvas.width = raw_width;
    newCanvas.height = raw_height;
    var staticCanvas = new fabric.StaticCanvas(newCanvas.id);
    staticCanvas.setWidth(raw_width);
    staticCanvas.setHeight(raw_height);
    staticCanvas.renderOnAddRemove = false;
    fabric.loadSVGFromURL('./output/' + prefix + '_raw.svg', function (objects, options) {
        var obj = fabric.util.groupSVGElements(objects, options);
        obj.selectable = false;
        obj.evented = false;
        obj.opacity = 1;
        staticCanvas.add(obj);
        // load uncertain usm
        var imgInstance = new fabric.Image(usmUnCanvas, {
            opacity: 1,
            selectable: false,
            evented: false
        });
        imgInstance.imageSmoothing = false;   // important for zooming
        staticCanvas.add(imgInstance); 
        staticCanvas.renderOnAddRemove = true;
        staticCanvas.renderAll();
        let level = 20;    // zoom before cropping, 20 is good enough
        staticCanvas.setViewportTransform([level, 0, 0, level, 0, 0]); 
        // copy regions
        for (var i = 0; i < usm_region_indices.length; i++) {
            copyUsmRegion(usm_region_indices[i], level, staticCanvas);
        }
    });
}

function copyUsmRegion(index, level, staticCanvas){
    let info = usm_region_dict[index];
    let bound = info["bound"];
    let l = bound[0], t = bound[1], w = bound[2], h = bound[3];
    // create new image
    var image = new Image();
    let coord = fabric.util.transformPoint({ x: l, y: t }, staticCanvas.viewportTransform); // position -> coordinate
    image.src =  staticCanvas.toDataURL({
        left: coord.x,
        top: coord.y,
        width: w * level,
        height: h * level
    });
    image.classList.add("region-img");
    image.style.width = (w *10) + "px";
    image.style.height = (h * 10) + "px";
    image.onclick = function() { 
        clickRegionImg(index); 
        log("click suggested USM region: " + index);
        usmRegionCnt++;
    };
    image.ondblclick = function() { 
        updateUncertainUsmRegion(index); 
        updateCanvasState();
        log("double click suggested USM region: " + index);
        usmRegionDblCnt++;
    };
    document.querySelector('#display-options').appendChild(image);
}


function clickRegionImg(index){
    let info = usm_region_dict[index];
    var bound = info["bound"];
    let center = info["center"];
    let wRatio = raw_width / bound[2];
    let hRatio = raw_height / bound[3];
    let zoom = hRatio > wRatio ? hRatio : wRatio;
    zoom = zoom/10;
    // recenter
    window.__canvas.setViewportTransform([ratio,0,0, ratio,0,0]); 
    // zoom
    let centerCoord = fabric.util.transformPoint({ x: raw_width/2, y: raw_height/2 }, window.__canvas.viewportTransform); // position -> coordinate
    let coord = fabric.util.transformPoint({ x: parseFloat(center[0]), y: parseFloat(center[1]) }, window.__canvas.viewportTransform); // position -> coordinate
    let offX = centerCoord.x - coord.x;
    let offY = centerCoord.y - coord.y;
    // zoom to make the region big enough
    window.__canvas.zoomToPoint(coord, zoom);     
    // translate the region center to frame center
    window.__canvas.viewportTransform[4] += offX;
    window.__canvas.viewportTransform[5] += offY;
    updateKpSize(zoom);  // update size
}

function updateCanvasState() { 
    var _canvasObject = window.__canvas;
    if((_config.redoFinishedStatus && _config.undoFinishedStatus)){
        var jsonData = _canvasObject.toJSON(['hasControls', 'selectable', 'strokeWidth', 'opacity', 'evented', 'visible', 'imageSmoothing']);
        var canvasAsJson = JSON.stringify(jsonData);
        var mgrState = window.manager.saveState();
        var usmSt = {"r":Array.from(usm_dirty_row), "c":Array.from(usm_dirty_column), 'd':Object.assign({}, usm_dirty_dict)};
        if(_config.currentStateIndex < _config.canvasState.length-1){
            var indexToBeInserted = _config.currentStateIndex+1;
            _config.canvasState[indexToBeInserted] = canvasAsJson;
            _config.managerState[indexToBeInserted] = mgrState;
            _config.usmState[indexToBeInserted] = usmSt;
            var numberOfElementsToRetain = indexToBeInserted+1;
            // discard following steps
            _config.canvasState = _config.canvasState.splice(0,numberOfElementsToRetain);
            _config.managerState = _config.managerState.splice(0,numberOfElementsToRetain);
            _config.usmState = _config.usmState.splice(0,numberOfElementsToRetain);
        }else{
            _config.canvasState.push(canvasAsJson);
            _config.managerState.push(mgrState);
            _config.usmState.push(usmSt);
        }
        _config.currentStateIndex = _config.canvasState.length-1;
        // update button status
        updateHistoryBtn();
        printLayers();
    }
}

function printLayers(){
    for (let i = 0; i < window.manager._layers.length; i++) {  
        // console.log("i:",i,", range: ", window.manager._layers[i].startIndex,",", window.manager._layers[i].endIndex);
    }
}

function printStat(){
    log(`Current stat. vecBtnCnt: ${vecBtnCnt}, updateBtnCnt:${updateBtnCnt}, finalBtnCnt:${finalBtnCnt}, undoBtnCnt:${undoBtnCnt}, redoBtnCnt:${redoBtnCnt}, deleteBtnCnt:${deleteBtnCnt}, lassoBtnCnt:${lassoBtnCnt}, penBtnCnt:${penBtnCnt}, eraseCnt:${eraseCnt}, rasterLayerCnt:${rasterLayerCnt}, rawLayerCnt:${rawLayerCnt}, updateLayerCnt:${updateLayerCnt}, finalLayerCnt:${finalLayerCnt}, kpLayerCnt:${kpLayerCnt}, unUsmLayerCnt:${unUsmLayerCnt}, usmLayerCnt:${usmLayerCnt}, usmRegionCnt:${usmRegionCnt}, usmRegionDblCnt:${usmRegionDblCnt}`);
}

function resetUsmPixels(usmState){
    let r_list=usmState['r'], c_list=usmState['c'], d_dict=usmState['d'];
    // reset all modified pixels
    for (var i = 0; i < usm_dirty_row.length; i++) {
        var pos = (usm_dirty_column[i] * raw_width + usm_dirty_row[i]) * 4;
        buffer[pos] = 0;           
        buffer[pos+1] = 0;           
        buffer[pos+2] = 0;           
        buffer[pos+3] = 0;
    }
    // recreate new pixels
    for (var i = 0; i < r_list.length; i++) {
        var pos = (c_list[i] * raw_width + r_list[i]) * 4;
        let key = 'r'+r_list[i]+'c'+c_list[i];
        if(d_dict[key]){
            buffer[pos] = COLOR_USM_R;           
            buffer[pos+1] = COLOR_USM_G;           
            buffer[pos+2] = COLOR_USM_B;           
            buffer[pos+3] = COLOR_USM_A;
        }
        else{
            buffer[pos] = 0;           
            buffer[pos+1] = 0;           
            buffer[pos+2] = 0;           
            buffer[pos+3] = 0;
        }
    }
    //reset current usm dirty status 
    usm_dirty_dict = Object.assign({}, d_dict);
}

function undo() {
    log("click undo button");
    undoBtnCnt++;
    var _canvasObject = window.__canvas;
    if(_config.undoFinishedStatus){
        if (_config.currentStateIndex > 0) {
            window.manager.enableEvt = false;
            _config.undoFinishedStatus = 0;
            if(_config.currentStateIndex != 0){
                _config.undoStatus = true;
                _canvasObject.renderOnAddRemove = false;
                // _canvasObject.clear();
                var jsonObj = JSON.parse(_config.canvasState[_config.currentStateIndex-1]);
                _canvasObject._objects = [];
                fabric.util.enlivenObjects(jsonObj.objects, function (enlivenedObjects) {
                    window.manager.restoreState(_config.managerState[_config.currentStateIndex-1]);
                    let usmIndex = window.manager.getLayer(LAYER_USM).startIndex;
                    // This is a dirty trick to solve the following strange issue: 
                    // object added through canvas.add() has a wrong index in canvas._objects array.
                    for (var i = 0; i < enlivenedObjects.length; i++) {
                        _canvasObject.add(enlivenedObjects[i]);
                    };
                    enlivenedObjects.forEach(function (obj, index) {
                        _canvasObject._objects[index] = obj;
                    }); 
                    resetUsmPixels(_config.usmState[_config.currentStateIndex-1]);
                    let oldUsm = _canvasObject._objects[usmIndex]; 
                    // set our buffer as source
                    idata.data.set(buffer);
                    // update canvas with new data
                    usmCtx.putImageData(idata, 0, 0); 
                    _canvasObject.remove(oldUsm); 
                    _canvasObject._objects.splice(usmIndex, 0, usmImg);
                    _canvasObject.renderAll();
                    _canvasObject.renderOnAddRemove = true;
                    _config.undoStatus = false;
                    _config.currentStateIndex -= 1; 
                    _config.undoFinishedStatus = 1;
                    window.manager.enableEvt = true;
                    updateHistoryBtn();
                    updateActiveLayer();
                    checkCheckers();
                });
            } 
        }
    }
}

function redo() {
    log("click redo button");
    redoBtnCnt++;
    var _canvasObject = window.__canvas;
    if(_config.redoFinishedStatus){
        if (_config.canvasState.length-1 > _config.currentStateIndex){
            window.manager.enableEvt = false;
            _config.redoFinishedStatus = 0;
            _config.redoStatus = true;
            _canvasObject.renderOnAddRemove = false;
            // _canvasObject.clear();
            var jsonObj = JSON.parse(_config.canvasState[_config.currentStateIndex+1]);
            _canvasObject._objects = [];
            fabric.util.enlivenObjects(jsonObj.objects, function (enlivenedObjects) {
                window.manager.restoreState(_config.managerState[_config.currentStateIndex+1]);
                let usmIndex = window.manager.getLayer(LAYER_USM).startIndex;
                // This is a dirty trick to solve the following strange issue: 
                // object added through canvas.add() has a wrong index in canvas._objects array.
                for (var i = 0; i < enlivenedObjects.length; i++) {
                    _canvasObject.add(enlivenedObjects[i]);
                };
                enlivenedObjects.forEach(function (obj, index) {
                    _canvasObject._objects[index] = obj;
                });
                resetUsmPixels(_config.usmState[_config.currentStateIndex+1]);
                let oldUsm = _canvasObject._objects[usmIndex]; 
                // set our buffer as source
                idata.data.set(buffer);
                // update canvas with new data
                usmCtx.putImageData(idata, 0, 0); 
                _canvasObject.remove(oldUsm);  
                _canvasObject._objects.splice(usmIndex, 0, usmImg);
                _canvasObject.renderAll();
                _canvasObject.renderOnAddRemove = true;
                _config.redoStatus = false;
                _config.currentStateIndex += 1;
                _config.redoFinishedStatus = 1;
                window.manager.enableEvt = true;
                updateHistoryBtn();
                updateActiveLayer();
                checkCheckers();
            });
        }
    }
};

function updateHistoryBtn(){
    // redo
    if(_config.currentStateIndex == _config.canvasState.length-1){
        $("#redo").prop('disabled', true);
    }
    else{
        if (_config.currentStateIndex < _config.canvasState.length-1){
            $("#redo").removeAttr('disabled');
        }
    }
    // undo
    if(_config.currentStateIndex > 0){
        $("#undo").removeAttr('disabled');
    }
    else{
        $("#undo").prop('disabled', true);
    }
}
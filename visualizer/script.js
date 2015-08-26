/***************/
/*** General ***/
/***************/

window.app = {};
var app = window.app;

app.mainLayer = new ol.layer.Tile({ source: new ol.source.OSM() });


/****************/
/*** Geometry ***/
/****************/

app.geometry = {}
app.geometry.REarth = 6371000;
app.geometry.toRadians = function(x){ return x * Math.PI / 180; };

app.geometry.haversine = function(lat1, lon1, lat2, lon2){
	var lat1 = app.geometry.toRadians(lat1);
	var lon1 = app.geometry.toRadians(lon1);
	var lat2 = app.geometry.toRadians(lat2);
	var lon2 = app.geometry.toRadians(lon2);

    var dlat = Math.abs(lat1-lat2);
    var dlon = Math.abs(lon1-lon2);

    var alpha = Math.pow(Math.sin(dlat/2), 2) + Math.cos(lat1) * Math.cos(lat2) * Math.pow(Math.sin(dlon/2), 2);
    var d = Math.atan2(Math.sqrt(alpha), Math.sqrt(1-alpha));

    return  2 * app.geometry.REarth * d;
};

app.geometry.equirectangular = function(lat1, lon1, lat2, lon2){
	var lat1 = app.geometry.toRadians(lat1);
	var lon1 = app.geometry.toRadians(lon1);
	var lat2 = app.geometry.toRadians(lat2);
	var lon2 = app.geometry.toRadians(lon2);
	var x = (lon2-lon1) * Math.cos((lat1+lat2)/2);
	var y = (lat2-lat1);
	return Math.sqrt(x*x + y*y) * app.geometry.REarth;
};


/***************/
/*** Measure ***/
/***************/

app.measure = {};
app.measure.tooltip_list = [];

app.measure.source = new ol.source.Vector();

app.measure.layer = new ol.layer.Vector({
	source: app.measure.source,
	style: new ol.style.Style({
		fill: new ol.style.Fill({
			color: 'rgba(255, 255, 255, 0.2)'
		}),
		stroke: new ol.style.Stroke({
			color: '#FC3',
			width: 2
		}),
		image: new ol.style.Circle({
			radius: 7,
			fill: new ol.style.Fill({
				color: '#FC3'
			})
		})
	})
});

app.measure.pointerMoveHandler = function(evt){
	if(evt.dragging){ return; }
	var tooltipCoord = evt.coordinate;

	if(app.measure.sketch){
		var output;
		var geom = (app.measure.sketch.getGeometry());
		if(geom instanceof ol.geom.LineString){
			output = app.measure.formatLength((geom));
			tooltipCoord = geom.getLastCoordinate();
		}
		app.measure.tooltipElement.innerHTML = output;
		app.measure.tooltip.setPosition(tooltipCoord);
	}
};

app.measure.addInteraction = function(){
	app.measure.draw = new ol.interaction.Draw({
		source: app.measure.source,
		type: ('LineString'),
		style: new ol.style.Style({
			fill: new ol.style.Fill({
				color: 'rgba(255, 255, 255, 0.2)'
			}),
			stroke: new ol.style.Stroke({
				color: 'rgba(0, 0, 0, 0.5)',
				lineDash: [10, 10],
				width: 2
			}),
			image: new ol.style.Circle({
				radius: 5,
				stroke: new ol.style.Stroke({
					color: 'rgba(0, 0, 0, 0.7)'
				}),
				fill: new ol.style.Fill({
					color: 'rgba(255, 255, 255, 0.2)'
				})
			})
		})
	});
	app.map.addInteraction(app.measure.draw);

	app.measure.createTooltip();

	app.measure.draw.on('drawstart',
		function(evt){
			app.measure.sketch = evt.feature;
		}, this);

	app.measure.draw.on('drawend',
		function(evt){
			app.measure.tooltipElement.className = 'measure-tooltip measure-tooltip-static';
			app.measure.tooltip.setOffset([0, -7]);
			app.measure.sketch = null;
			app.measure.tooltipElement = null;
			app.measure.createTooltip();
		}, this);
};

app.measure.createTooltip = function(){
	if(app.measure.tooltipElement){
		app.measure.tooltipElement.parentNode.removeChild(app.measure.tooltipElement);
	}
	app.measure.tooltipElement = document.createElement('div');
	app.measure.tooltipElement.className = 'measure-tooltip measure-tooltip-value';
	app.measure.tooltip = new ol.Overlay({
		element: app.measure.tooltipElement,
		offset: [0, -15],
		positioning: 'bottom-center'
	});
	app.measure.tooltip_list.push(app.measure.tooltip);
	app.map.addOverlay(app.measure.tooltip);
};

app.measure.formatLength = function(line){
	var length_euclidean = line.getLength();
	var length_equirectangular = 0;
	var length_haversine = 0;
	var coordinates = line.getCoordinates();
	var sourceProj = app.map.getView().getProjection();
	for(var i = 0, ii = coordinates.length - 1; i < ii; ++i){
		var c1 = ol.proj.transform(coordinates[i], sourceProj, 'EPSG:4326');
		var c2 = ol.proj.transform(coordinates[i + 1], sourceProj, 'EPSG:4326');
		length_equirectangular += app.geometry.equirectangular(c1[1], c1[0], c2[1], c2[0]);
		length_haversine += app.geometry.haversine(c1[1], c1[0], c2[1], c2[0]);
	}

	var disp = function(x){
		if(x > 100){
			return Math.round(x / 1000 * 1000) / 1000 + 'km';
		} else {
			return Math.round(x * 1000) / 1000 + 'm';
		}
	}

	var length_euclidean = disp(length_euclidean);
	var length_equirectangular = disp(length_equirectangular);
	var length_haversine = disp(length_haversine);

	var display_euclidean = $('input#measure-euclidean').prop('checked');
	var display_equirectangular = $('input#measure-equirectangular').prop('checked');
	var display_haversine = $('input#measure-haversine').prop('checked');

	var header = true;
	if(display_euclidean + display_equirectangular + display_haversine == 1){
		header = false;
	}

	var str = '';
	if(display_euclidean){
		if(header){ str += 'euclidean: '; }
		str += length_euclidean;
	}
	if(display_equirectangular){
		if(header){ if(display_euclidean){ str += '<br>'; } str += 'equirectangular: '; }
		str += length_equirectangular;
	}
	if(display_haversine){
		if(header){ str += '<br> haversine: '; }
		str += length_haversine;
	}
	return str;
};


/*******************/
/*** DataDisplay ***/
/*******************/

app.dataDisplay = {};
app.dataDisplay.layers = {};
app.dataDisplay.heatmapRadius = 5;
app.dataDisplay.heatmapBlur = 5;
app.dataDisplay.pathPointMode = 1; // endpoints
app.dataDisplay.pathPointResolution = 50;

app.dataDisplay.loadLayer = function(path){
	$.ajax({url: path, cache: false, dataType: 'json',
		success: function(result){
			app.dataDisplay.layers[path] = app.dataDisplay.preprocess(result);
			app.map.addLayer(app.dataDisplay.layers[path]);
		}
	});
};

app.dataDisplay.unloadLayer = function(path){
	app.map.removeLayer(app.dataDisplay.layers[path]);
	delete app.dataDisplay.layers[path];
};

app.dataDisplay.rawStyle = function(feature, resolution){
	var style = [ new ol.style.Style({
			stroke: new ol.style.Stroke({
				color: '#00F',
				width: 5
			}),
			image: new ol.style.Circle({
				radius: 5,
				fill: new ol.style.Fill({
					color: '#00F'
				})
			})
		}),
		new ol.style.Style({
			stroke: new ol.style.Stroke({
				color: '#000',
				width: 2
			}),
			image: new ol.style.Circle({
				radius: 2,
				fill: new ol.style.Fill({
					color: '#FFF'
				})
			})
		})
	];

	if(feature.get('display') == 'path' && resolution < app.dataDisplay.pathPointResolution){
		if(app.dataDisplay.pathPointMode == 2){
			var polyline = feature.getGeometry();
			var points = polyline.getCoordinates();
			for(var i=1; i<points.length-1; ++i){
				var point = points[i];
				var pos = i/points.length;
				var red = 0;
				var green = 0;
				if(pos < 0.5){
					green = 255;
					red = Math.round(pos*2*255);
				} else {
					red = 255;
					green = Math.round((1-pos)*2*255);
				}
				style.push(new ol.style.Style({
					geometry: new ol.geom.Point(point),
					image: new ol.style.Circle({
						radius: 3,
						fill: new ol.style.Fill({
							color: 'rgb('+red+','+green+',0)'
						})
					})
				}));
			}
		}
		if(app.dataDisplay.pathPointMode >= 1){
			var polyline = feature.getGeometry();
			var first = polyline.getFirstCoordinate();
			var last = polyline.getLastCoordinate();
			style.push(new ol.style.Style({
				geometry: new ol.geom.Point(first),
				image: new ol.style.Circle({
					radius: 5,
					fill: new ol.style.Fill({
						color: '#0F0'
					})
				})
			}));
			style.push(new ol.style.Style({
				geometry: new ol.geom.Point(last),
				image: new ol.style.Circle({
					radius: 5,
					fill: new ol.style.Fill({
						color: '#F00'
					})
				})
			}));
		}
	}

	return style;
};

app.dataDisplay.clusterStyleCache = {};
app.dataDisplay.clusterStyle = function(feature, resolution){
	var size = feature.get('features').length;
	var style = app.dataDisplay.clusterStyleCache[size];
	if(!style){
		style = [new ol.style.Style({
			image: new ol.style.Circle({
				radius: 10,
				stroke: new ol.style.Stroke({
					color: '#FFF'
				}),
				fill: new ol.style.Fill({
					color: '#39C'
				})
			}),
			text: new ol.style.Text({
				text: size.toString(),
				fill: new ol.style.Fill({
					color: '#FFF'
				})
			})
		})];
		app.dataDisplay.clusterStyleCache[size] = style;
	}
	return style;
};

app.dataDisplay.pointDistributionStyle = function(feature, resolution){
	var p = feature.get('info');
	var red = 0;
	var green = 0;
	if(p < 0.5){
		green = 255;
		red = Math.round(p*2*255);
	} else {
		red = 255;
		green = Math.round((1-p)*2*255);
	}
	return [ new ol.style.Style({
		image: new ol.style.Circle({
			radius: 5,
			fill: new ol.style.Fill({
				color: 'rgb('+red+','+green+',0)'
			})
		})
	}) ];
};

app.dataDisplay.preprocess = function(egj){
	var source = new ol.source.GeoJSON({
		projection: 'EPSG:3857',
		object: egj.data
	});

	if(egj.type == 'raw'){
		return new ol.layer.Vector({
			source: source,
			style: app.dataDisplay.rawStyle
		});

	} else if(egj.type == 'cluster'){
		return new ol.layer.Vector({
			source: new ol.source.Cluster({
				distance: 40,
				source: source
			}),
			style: app.dataDisplay.clusterStyle
		});

	} else if(egj.type == 'heatmap'){
		return new ol.layer.Heatmap({
			source: source,
			blur: app.dataDisplay.heatmapBlur,
			radius: app.dataDisplay.heatmapRadius
		});
	} else if(egj.type == 'point distribution'){
		return new ol.layer.Vector({
			source: source,
			style: app.dataDisplay.pointDistributionStyle
		});
	}
};

app.dataDisplay.reloadPathes = function(){
	for(var layer in app.dataDisplay.layers){
		if(app.dataDisplay.layers[layer].getSource().getFeatures()[0].get('display') == 'path'){
			app.dataDisplay.layers[layer].changed();
		}
	}
};

app.dataDisplay.reloadHeatmaps = function(){
	for(var key in app.dataDisplay.layers){
		var layer = app.dataDisplay.layers[key];
		if(layer instanceof ol.layer.Heatmap){
			layer.setBlur(app.dataDisplay.heatmapBlur);
			layer.setRadius(app.dataDisplay.heatmapRadius);
		}
	}
};


/****************/
/*** DataList ***/
/****************/

app.dataList = {};
app.dataList.current = {};
app.dataList.idgen = 0;

app.dataList.init = function(){
	app.dataList.elementTree = {};
	app.dataList.elementTree.parent = null;
	app.dataList.elementTree.children = {};
	app.dataList.elementTree.checkbox = null;
	app.dataList.elementTree.ul = $('#datalist-tree ul');

	app.dataList.updateList();
	setInterval(app.dataList.updateList, 1000);
};

app.dataList.updateList = function(){
	$.ajax({url: '/ls/', cache: false, dataType: 'json',
		success: function(result){
			result.forEach(function(file){
				file.uri = file.path.join('/') + '/' + file.name
				if(file.uri in app.dataList.current){
					if(file.mtime > app.dataList.current[file.uri].mtime){
						var act = app.dataList.current[file.uri];
						if(act.checkbox.prop('checked')){
							app.dataList.unloadLayer(file.uri);
							app.dataList.loadLayer(file.uri);
						}
						act.mtime = file.mtime;
					}
				} else {
					app.dataList.insert(file);
				}
			});
		}
	});
};

app.dataList.insert = function(file){
	var cur = app.dataList.elementTree;
	var prev = null;
	for(var i = 1; i<file.path.length; i++){
		if(!(file.path[i] in cur.children)){
			var n = {};
			n.uri = file.path.slice(0, i+1).join('/');
			n.children = {};
			n.parent = cur;
			n.ul = $('<ul>')
				.prop('id', 'folder-'+app.dataList.idgen)
				.hide();

			var hidelink = $('<a>')
				.prop('href', '')
				.append('hide')
				.hide();
			var showlink = $('<a>')
				.prop('href', '')
				.append('show');

			var playlink = $('<a>')
				.prop('href', '')
				.append('play');
			var stoplink = $('<a>')
				.prop('href', '')
				.append('stop')
				.hide();

			n.checkbox = $('<input>')
				.prop('type', 'checkbox')
				.prop('id', 'data-'+app.dataList.idgen)
				.prop('name', n.uri);
			n.checkbox.change(app.dataList.selectData);
			var item = $('<li>')
				.append(n.checkbox)
				.append($('<label>')
					.prop('for', 'data-'+app.dataList.idgen)
					.append(file.path[i]))
				.append(' ')
				.append(hidelink)
				.append(showlink)
				.append(' ')
				.append(playlink)
				.append(stoplink)
				.append(n.ul)
			app.dataList.idgen++;
			cur.ul.append(item);
			cur.children[file.path[i]] = n;
			app.dataList.current[n.uri] = n;

			var foldertoggler = function(){
				hidelink.toggle();
				showlink.toggle();
				n.ul.toggle();
				return false;
			};
			hidelink.click(foldertoggler);
			showlink.click(foldertoggler);

			playlink.click(function(){
				playlink.toggle();
				stoplink.toggle();
				app.dataPlayer.play(n);
				return false;
			});
			stoplink.click(function(){
				playlink.toggle();
				stoplink.toggle();
				app.dataPlayer.stop(n);
				return false;
			});
		}
		prev = cur;
		cur = cur.children[file.path[i]];
	}

	file.parent = cur;
	file.checkbox = $('<input>')
		.prop('type', 'checkbox')
		.prop('id', 'data-'+app.dataList.idgen)
		.prop('name', file.uri);
	file.checkbox.change(app.dataList.selectData);
	var item = $('<li>')
		.append(file.checkbox)
		.append($('<label>')
			.prop('for', 'data-'+app.dataList.idgen)
			.append(file.name))
	app.dataList.idgen++;
	cur.ul.append(item);
	cur.children[file.name] = file;
	app.dataList.current[file.uri] = file;

	if(cur.checkbox && cur.checkbox.prop('checked')){
		file.checkbox.prop('checked', true);
		app.dataList.updateData(file);
	}
};

app.dataList.updateData = function(cur){
	if(cur.checkbox.prop('checked')){
		app.dataList.loadLayer(cur.uri);
	} else {
		app.dataList.unloadLayer(cur.uri);
	}
};

app.dataList.updateCheckboxes = function(cur){
	if(cur.checkbox.prop('checked')){
		app.dataList.check(cur);
	} else {
		app.dataList.uncheck(cur);
	}
};

app.dataList.selectData = function(e){
	var cur = app.dataList.current[e.target.name];
	if(!('children' in cur)){
		app.dataList.updateData(cur);
	}
	app.dataList.updateCheckboxes(cur);
};

app.dataList.changeChildren = function rec(cur, val){
	cur.checkbox.prop('checked', val);
	if('children' in cur){
		for(var child in cur.children){
			rec(cur.children[child], val);
		}
	} else {
		app.dataList.updateData(cur);
	}
};

app.dataList.check = function(cur){
	// Check all parents
	var p = cur.parent;
	while(p.checkbox != null){
		p.checkbox.prop('checked', true);
		p = p.parent;
	}

	// Check all children
	for(var child in cur.children){
		app.dataList.changeChildren(cur.children[child], true);
	}
};

app.dataList.uncheck = function(cur){
	// Uncheck empty parents
	var p = cur.parent;
	while(p.checkbox != null && p.checkbox.prop('checked')){
		var cc = false;
		for(var child in p.children){
			cc = cc || p.children[child].checkbox.prop('checked');
		}
		if(cc){
			break;
		}
		p.checkbox.prop('checked', false);
		p = p.parent;
	}

	// Uncheck all children
	for(var child in cur.children){
		app.dataList.changeChildren(cur.children[child], false);
	}
};


app.dataList.loadLayer = function(uri){
	app.dataDisplay.loadLayer('/get'+uri);
};

app.dataList.unloadLayer = function(uri){
	app.dataDisplay.unloadLayer('/get'+uri);
};


/*******************/
/*** DataExtract ***/
/*******************/

app.dataExtract = {};
app.dataExtract.current = null;

app.dataExtract.init = function(){
	$('#dataextract button:contains("Refresh")').click(app.dataExtract.display);
	$('#dataextract input').keypress(function(e){
		if(e.keyCode == 13){
			app.dataExtract.display();
		}
	});
	$('#dataextract button:contains("Clear")').click(app.dataExtract.clear);
};

app.dataExtract.display = function(){
	if(app.dataExtract.current){
		app.dataDisplay.unloadLayer('/extract/' + app.dataExtract.current);
	}
	app.dataExtract.current = $('#dataextract input').val();
	app.dataDisplay.loadLayer('/extract/' + app.dataExtract.current);
};

app.dataExtract.clear = function(){
	if(app.dataExtract.current){
		app.dataDisplay.unloadLayer('/extract/' + app.dataExtract.current);
		app.dataExtract.current = null;
	}
};



/******************/
/*** DataPlayer ***/
/******************/

app.dataPlayer = {};
app.dataPlayer.current = {};
app.dataPlayer.updateFrequency = 200;
app.dataPlayer.time = 0;

app.dataPlayer.init = function(){
	app.dataPlayer.intervalId = setInterval(app.dataPlayer.update, app.dataPlayer.updateFrequency);
};

app.dataPlayer.updateInterval = function(){
	clearInterval(app.dataPlayer.intervalId);
	app.dataPlayer.intervalId = setInterval(app.dataPlayer.update, app.dataPlayer.updateFrequency);
};

app.dataPlayer.play = function(cur){
	app.dataPlayer.updateKeys(cur);
	if(cur.keys.length == 0){
		alert("ERROR: No number in directory.");
		return;
	}
	app.dataList.uncheck(cur);
	cur.checkbox.prop('checked', true);
	cur.playIndex = 0;
	app.dataPlayer.current[cur.uri] = cur;
	for(var key in cur.context){
		var child = cur.children[cur.context[key]];
		child.checkbox.prop('checked', true);
		if(!('children' in child)){
			app.dataList.updateData(child);
		}
		app.dataList.updateCheckboxes(child);
	}
};

app.dataPlayer.updateKeys = function(cur){
	var keys = Object.keys(cur.children);
	cur.keys = keys.map(Number).filter(function(x){ return !isNaN(x); }).sort(function(l,r){return l-r;});
	cur.context = keys.filter(function(x){ return isNaN(Number(x)); });
};

app.dataPlayer.stop = function(cur){
	delete app.dataPlayer.current[cur.uri];
	delete cur.keys;
	delete cur.context;
	delete cur.playIndex;
};

app.dataPlayer.update = function(){
	for(var key in app.dataPlayer.current){
		var cur = app.dataPlayer.current[key];
		var prev = cur.children[cur.keys[cur.playIndex]];
		cur.playIndex++;
		if(cur.playIndex >= cur.keys.length){
			app.dataPlayer.updateKeys(cur);
			if(cur.playIndex >= cur.keys.length){
				cur.playIndex = 0;
			}
		}
		var next = cur.children[cur.keys[cur.playIndex]];

		prev.checkbox.prop('checked', false);
		app.dataList.updateData(prev);
		next.checkbox.prop('checked', true);
		app.dataList.updateData(next);
	}
};


/*****************/
/*** CoordInfo ***/
/*****************/

app.coordInfo = {};
app.coordInfo.init = function(){
	app.coordInfo.element = $('#coordInfo');
	app.coordInfo.enable();
};

app.coordInfo.enable = function(){
	app.map.on('pointermove', app.coordInfo.update);
	app.coordInfo.element.show();
};

app.coordInfo.disable = function(){
	app.coordInfo.element.hide();
	app.map.un('pointermove', app.coordInfo.update);
};

app.coordInfo.update = function(evt){
	var coord = ol.proj.transform(app.map.getEventCoordinate(evt.originalEvent), app.map.getView().getProjection(), 'EPSG:4326');
	app.coordInfo.element.text(coord[1] + ', ' + coord[0]);
};


/*******************/
/*** FeatureInfo ***/
/*******************/

app.featureInfo = {};

app.featureInfo.init = function(){
	app.featureInfo.element = $('#featureinfo');
	app.featureInfo.static = $('#featureinfo-static');
	app.featureInfo.dynamic = $('#featureinfo-dynamic');
	app.featureInfo.element.hide();
	app.featureInfo.enable();
};

app.featureInfo.enable = function(){
	app.map.on('pointermove', app.featureInfo.updateMove);
	app.map.on('click', app.featureInfo.updateClick);
};

app.featureInfo.disable = function(){
	app.map.un('pointermove', app.featureInfo.updateMove);
	app.map.un('click', app.featureInfo.updateClick);
};

app.featureInfo.updateMove = function(evt){
	if(evt.dragging){
		app.featureInfo.element.hide();
		return;
	}
	app.featureInfo.display(evt);
};

app.featureInfo.updateClick = function(evt){
	app.featureInfo.display(evt);
};

app.featureInfo.display = function(evt){
	app.featureInfo.element.css({
		left: (evt.pixel[0] + 10) + 'px',
		top: evt.pixel[1] + 'px'
	});
	var feature = app.map.forEachFeatureAtPixel(evt.pixel, function(feature, layer) {
		return feature;
	});
	if(feature && feature.get('info')){
		if(feature.get('display') == 'path'){
			var dtime = app.featureInfo.interpolateTime(feature.getGeometry(), evt.coordinate);
			var date = new Date(1000*(feature.get('timestamp') + dtime*15));
			var desc = 'index: '+dtime+'<br>';
			desc += 'date: '+date.toISOString()+'<br>';
			app.featureInfo.dynamic.html(desc);
		} else {
			app.featureInfo.dynamic.html('');
		}
		app.featureInfo.static.html(feature.get('info'));
		app.featureInfo.element.show();
	} else {
		app.featureInfo.element.hide();
	}
};

app.featureInfo.interpolateTime = function(polyline, coord){
	var closest = polyline.getClosestPoint(coord);
	var best = 1000;
	var bestStart = -1;
	var bestEnd = -1;
	var bestI = -1;
	var i = 0;
	var points = polyline.getCoordinates();
	for(var i=0; i<points.length-1; i++){
		var start = points[i];
		var end = points[i+1];
		var dist = Math.abs((end[0]-start[0])*closest[1] - (end[1]-start[1])*closest[0] + end[1]*start[0] - end[0]*start[1]) / Math.sqrt(Math.pow(end[0]-start[0], 2) + Math.pow(end[1]-start[1], 2));
		if(dist<best){
			best = dist;
			bestStart = start;
			bestEnd = end;
			bestI = i;
		}
	}

	if(bestI == -1){
		return 0;
	}

	var distClosest = app.geometry.equirectangular(bestStart[1], bestStart[0], closest[1], closest[0]);
	var distEnd = app.geometry.equirectangular(bestStart[1], bestStart[0], bestEnd[1], bestEnd[0]);
	var ratio = distClosest / distEnd;
	return bestI + ratio;
};


/***************/
/*** Control ***/
/***************/

app.control = {};

app.control.OpenConfigControl = function(opt_options){
	var options = opt_options || {};
	
	var button = document.createElement('button');
	button.innerHTML = '⚙';
	
	button.addEventListener('click', function(e){ $('#config').toggle() }, false);
	button.addEventListener('touchstart', function(e){ $('#config').toggle() }, false);
	
	var element = document.createElement('div');
	element.className = 'open-config ol-unselectable ol-control';
	element.appendChild(button);
	
	ol.control.Control.call(this, {
		element: element,
		target: options.target
	});
};
ol.inherits(app.control.OpenConfigControl, ol.control.Control);

app.control.OpenDatalist = function(opt_options){
	var options = opt_options || {};
	
	var button = document.createElement('button');
	button.innerHTML = '«';
	
	var toggler = function(e){
		$('#datalist').toggle();
		if(button.innerHTML == '«'){
			button.innerHTML = '»';
			$('.open-datalist').css('right', '20.5em');
		}
		else{
			button.innerHTML = '«';
			$('.open-datalist').css('right', '.5em');
		}
	};
	button.addEventListener('click', toggler, false);
	button.addEventListener('touchstart', toggler, false);
	
	var element = document.createElement('div');
	element.className = 'open-datalist ol-unselectable ol-control';
	element.appendChild(button);
	
	ol.control.Control.call(this, {
		element: element,
		target: options.target
	});
};
ol.inherits(app.control.OpenDatalist, ol.control.Control);


/************/
/*** Menu ***/
/************/

app.menu = {};

app.menu.init = function(){
	$('#config ul').menu();

	$('#config ul li').click(function(e){
		switch($(this).text()){
			case 'Enable coord':
				app.coordInfo.enable();
				$('#config ul li:contains("Enable coord")').toggle();
				$('#config ul li:contains("Disable coord")').toggle();
				break;
			case 'Disable coord':
				app.coordInfo.disable();
				$('#config ul li:contains("Enable coord")').toggle();
				$('#config ul li:contains("Disable coord")').toggle();
				break;
			case 'Set player speed':
				var tmp = prompt("Player update frequency (milliseconds)", app.dataPlayer.updateFrequency);
				if(tmp){
					app.dataPlayer.updateFrequency = parseInt(tmp);
					app.dataPlayer.updateInterval();
				}
				break;
		}
	});

	$('ul#config-measure li').click(function(e){
		switch($(this).text()){
			case 'Enable':
				app.measure.addInteraction();
				app.map.on('pointermove', app.measure.pointerMoveHandler);
				app.featureInfo.disable();
				$('ul#config-measure li:contains("Enable")').toggle();
				$('ul#config-measure li:contains("Disable")').toggle();
				break;
			case 'Disable':
				app.featureInfo.enable();
				app.map.un('pointermove', app.measure.pointerMoveHandler);
				app.map.removeInteraction(app.measure.draw);
				app.measure.draw = null;
				$('ul#config-measure li:contains("Enable")').toggle();
				$('ul#config-measure li:contains("Disable")').toggle();
				break;
			case 'Clear':
				app.measure.source.clear();
				app.measure.tooltip_list.forEach(function(e){
					app.map.removeOverlay(e);
				});
				app.measure.tooltip_list.length = 0;
				if(app.measure.draw){
					app.map.removeInteraction(app.measure.draw);
					app.measure.addInteraction();
				}
				break;
		}
	});

	$('ul#config-layer li').click(function(e){
		switch($(this).text()){
			case 'OSM':
				app.mainLayer = new ol.layer.Tile({ source: new ol.source.OSM() });
				break;
			case 'Bing':
				app.mainLayer = new ol.layer.Tile({ source: new ol.source.BingMaps({
					key: 'Ak-dzM4wZjSqTlzveKz5u0d4IQ4bRzVI309GxmkgSVr1ewS6iPSrOvOKhA-CJlm3',
					imagerySet: 'AerialWithLabels',
					maxZoom: 19
				}) });
				break;
			case 'Bing (no labels)':
				app.mainLayer = new ol.layer.Tile({ source: new ol.source.BingMaps({
					key: 'Ak-dzM4wZjSqTlzveKz5u0d4IQ4bRzVI309GxmkgSVr1ewS6iPSrOvOKhA-CJlm3',
					imagerySet: 'Aerial',
					maxZoom: 19
				}) });
				break;
		}
		app.map.getLayers().setAt(0, app.mainLayer);
	});

	$('ul#config-pathdraw li').click(function(e){
		switch($(this).text()){
			case 'Set resolution':
				var tmp = prompt("Path points resolution", app.dataDisplay.pathPointResolution);
				if(tmp){
					app.dataDisplay.pathPointResolution = parseInt(tmp);
					app.dataDisplay.reloadPathes();
				}
				break;
			case 'No points':
				app.dataDisplay.pathPointMode = 0;
				app.dataDisplay.reloadPathes();
				break;
			case 'Endpoints':
				app.dataDisplay.pathPointMode = 1;
				app.dataDisplay.reloadPathes();
				break;
			case 'All points':
				app.dataDisplay.pathPointMode = 2;
				app.dataDisplay.reloadPathes();
				break;
		}
	});

	$('ul#config-heatmap li').click(function(e){
		switch($(this).text()){
			case 'Set blur':
				var tmp = prompt("Heatmap blur", app.dataDisplay.heatmapBlur);
				if(tmp){
					app.dataDisplay.heatmapBlur = parseInt(tmp);
					app.dataDisplay.reloadHeatmaps();
				}
				break;
			case 'Set radius':
				var tmp = prompt("Heatmap radius", app.dataDisplay.heatmapRadius);
				if(tmp){
					app.dataDisplay.heatmapRadius = parseInt(tmp);
					app.dataDisplay.reloadHeatmaps();
				}
				break;
		}
	});
};


/**********************/
/*** Initialization ***/
/**********************/

$(function(){
	app.map = new ol.Map({
		controls: ol.control.defaults().extend([
			new app.control.OpenConfigControl(),
			new app.control.OpenDatalist()
		]),
		target: 'map',
		layers: [ app.mainLayer, app.measure.layer ],
		view: new ol.View({
			center: ol.proj.transform([-8.621953, 41.162142], 'EPSG:4326', 'EPSG:3857'),
			zoom: 13
		})
	});

	app.menu.init();
	app.coordInfo.init();
	app.featureInfo.init();
	app.dataList.init();
	app.dataExtract.init();
	app.dataPlayer.init();
});

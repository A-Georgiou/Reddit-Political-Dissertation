import { useEffect } from "react"
import * as d3 from "d3";
import React from 'react';
import * as venn from "venn.js";

const VennDiagram = (props) => {
    const chartView = React.useRef(null);
    let chart = venn.VennDiagram().fontSize("24px");
    const width = 900;
    const height = 800;
    useEffect(() => {
        let div = d3.select(chartView.current);
        const { vennData } = props;
        div.datum(vennData).call(chart.height(height).width(width));
        
        const vennDiv = document.getElementById("venn");
        const vennSvg = vennDiv.children[0];

        vennDiv.setAttribute("class", "svg-container oneten-height");
        vennSvg.removeAttribute("height");
        vennSvg.removeAttribute("width");
        vennSvg.removeAttribute("viewBox");
        vennSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        vennSvg.setAttribute("preserveAspectRatio", "xMaxYMin meet");
        vennSvg.setAttribute("class", "svg-content-responsive");

        div.selectAll("g")
        .on("mouseover", function (event, d, i) {
            venn.sortAreas(div, d);
            var node = d3.select(this).transition();
            if(d.sets[0].toLowerCase() !== props.current.toLowerCase() || d.sets.length > 1){
                d3.select(this).style("cursor", "pointer");
                node.select("path").style("fill-opacity", .2);
                node.select("text").style("font-weight", "bold").style("font-size", "36px");

                var selection = d3.select(this).transition().duration(400);
                selection.select("path")
                    .style("stroke-width", 3)
                    .style("fill-opacity", d.sets.length === 1 ? .4 : .1)
                    .style("stroke-opacity", 1);
            }

            if(d.sets.length > 1){
                var curr = div.selectAll("g.venn-area.venn-circle")._groups[0]
                for(let i = 0; i < curr.length; i++){
                    let currentG = curr[i];
                    if(currentG.dataset.vennSets.toLowerCase() === d.sets[1].toLowerCase()){
                        var current = d3.select(currentG).transition();
                        current.select("text").style("font-weight", "bold").style("font-size", "36px");
                    }
                }
            }

            
        })
        
        .on("mouseout", function (event, d, i) {
            d3.select(this).style("cursor", "default");
            var node = d3.select(this).transition();
            if(d.sets[0].toLowerCase() !== props.current.toLowerCase() || d.sets.length > 1){
                node.select("path").style("fill-opacity", 0);
                node.select("text").style("font-weight", "100").style("font-size", "24px");

                var selection = d3.select(this).transition().duration(400);
                selection.select("path")
                .style("stroke-width", 0)
                .style("fill-opacity", d.sets.length === 1 ? .25 : .0)
                .style("stroke-opacity", 0);
            }

            if(d.sets.length > 1){
                var curr = div.selectAll("g.venn-area.venn-circle")._groups[0]
                for(let i = 0; i < curr.length; i++){
                    let currentG = curr[i];
                    if(currentG.dataset.vennSets.toLowerCase() === d.sets[1].toLowerCase()){
                        var current = d3.select(currentG).transition();
                        current.select("text").style("font-weight", "100").style("font-size", "24px");
                    }
                }
            }            
        })

        .on("click", function(event, d) {
            if(d.sets[0].toLowerCase() !== props.current.toLowerCase() || d.sets.length > 1){
                var searchParameter = d.sets[d.sets.length-1];
                props.searchVenn(searchParameter);
            }
        })
    }, [props.vennData, chartView.current]);

    return(
        <div className="venn-div">
            <div className="" id="venn" ref={chartView}></div>
        </div>
    )

}

export default VennDiagram
// Vendored entry point for the Memory Inspector pipeline tab.
//
// Re-exports the subset of React + @xyflow/react that the inspector uses.
// Add new exports here when extending the pipeline UI; rebuild via
// `npm run build` and commit the resulting xyflow-bundle.{js,css}.

import * as React from "react";
import * as ReactDOMClient from "react-dom/client";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  useReactFlow,
  MarkerType,
} from "@xyflow/react";
import * as dagre from "@dagrejs/dagre";

export {
  React,
  ReactDOMClient,
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  useReactFlow,
  MarkerType,
  dagre,
};

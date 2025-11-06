import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    'installation',
    'getting-started',
    {
      type: 'category',
      label: 'Models',
      items: [
        'models/generic',
        'models/photosynthesis',
        'models/stomatal-conductance',
        'models/hydraulics',
        'models/canopy',
      ],
    },
    {
      type: 'category',
      label: 'Utilities',
      items: [
        'utilities/li600-correction',
      ],
    },
    {
      type: 'category',
      label: 'Community Notebooks',
      items: [
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/index',
      ],
    },
  ],
};

export default sidebars;

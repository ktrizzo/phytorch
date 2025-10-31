import type {ReactNode} from 'react';
import {useState} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText('pip install phytorch');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <img
          src="/img/logo.png"
          alt="PhyTorch Logo"
          style={{
            height: '280px',
            marginBottom: '1.5rem'
          }}
        />
        <Heading as="h1" className="hero__title" style={{
          fontSize: '4rem',
          fontWeight: 300,
          marginBottom: '1rem'
        }}>
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle" style={{fontSize: '1.5rem', marginBottom: '2.5rem'}}>
          {siteConfig.tagline}
        </p>
        <div className={styles.buttons} style={{gap: '1rem', marginBottom: '2.5rem'}}>
          <Link
            className="button button--outline button--primary button--lg"
            to="/docs/intro">
            Introduction
          </Link>
          <Link
            className="button button--outline button--primary button--lg"
            to="/docs/intro">
            Get started
          </Link>
          <Link
            className="button button--outline button--primary button--lg"
            to="/docs/intro">
            Tutorials
          </Link>
        </div>
        <div style={{
          marginTop: '2rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '0.5rem'
        }}>
          <pre style={{
            display: 'inline-block',
            padding: '1rem 2rem',
            backgroundColor: 'rgba(0, 0, 0, 0.1)',
            borderRadius: '8px',
            fontSize: '1.1rem',
            border: '1px solid rgba(0, 0, 0, 0.1)',
            margin: 0
          }}>
            <code>pip install phytorch</code>
          </pre>
          <button
            onClick={copyToClipboard}
            title={copied ? 'Copied!' : 'Copy to clipboard'}
            style={{
              padding: '0.75rem',
              backgroundColor: 'var(--copy-button-bg)',
              color: 'var(--copy-button-color)',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'opacity 0.2s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.opacity = '0.8'}
            onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
          >
            {copied ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12"></polyline>
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            )}
          </button>
        </div>
      </div>
    </header>
  );
}

function ReferencesSection() {
  return (
    <section style={{ padding: '4rem 0', textAlign: 'center' }}>
      <div className="container">
        <h2 style={{ fontSize: '2.5rem', marginBottom: '2rem' }}>References</h2>
        <div style={{ textAlign: 'left', maxWidth: '900px', margin: '0 auto' }}>
          <Link
            to="/docs/intro"
            style={{
              fontSize: '1.1rem',
              textDecoration: 'none',
              display: 'block',
              marginBottom: '1.5rem'
            }}>
            PhyTorch: A Framework for Physiological Plant Modeling
          </Link>
          <pre className="code-block-bg" style={{
            padding: '1.5rem',
            borderRadius: '8px',
            fontSize: '0.9rem',
            lineHeight: '1.6',
            overflow: 'auto',
            border: '1px solid var(--code-border-light)'
          }}>
            <code>{`@article{phytorch2024,
  title = {{PhyTorch: A Framework for Physiological Plant Modeling}},
  author = {[Authors]},
  journal = {[Journal]},
  year = {2024},
  url = {https://phytorch.org}
}`}</code>
          </pre>
          <p style={{ marginTop: '1.5rem', fontSize: '1rem' }}>
            Based on <Link to="/docs/intro">PhoTorch</Link> for photosynthesis modeling.
          </p>
        </div>
      </div>
    </section>
  );
}

function GetStartedSection() {
  return (
    <section style={{
      padding: '4rem 0',
      backgroundColor: 'var(--ifm-background-color)',
      textAlign: 'center'
    }}>
      <div className="container">
        <h2 style={{ fontSize: '2.5rem', marginBottom: '3rem' }}>Get Started</h2>
        <div style={{ textAlign: 'left', maxWidth: '900px', margin: '0 auto' }}>
          <div style={{ marginBottom: '3rem' }}>
            <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem' }}>
              <strong>1. Install PhyTorch:</strong>
            </h3>
            <p style={{ marginBottom: '1rem', color: 'var(--ifm-color-emphasis-700)' }}>
              via pip (recommended):
            </p>
            <pre className="code-block-bg" style={{
              padding: '1rem 1.5rem',
              borderRadius: '8px',
              fontSize: '1rem',
              border: '1px solid var(--code-border-light)',
              marginBottom: '1.5rem'
            }}>
              <code>pip install phytorch</code>
            </pre>
            <p style={{ marginBottom: '1rem', color: 'var(--ifm-color-emphasis-700)' }}>
              via Anaconda (from the unofficial conda-forge channel):
            </p>
            <pre className="code-block-bg" style={{
              padding: '1rem 1.5rem',
              borderRadius: '8px',
              fontSize: '1rem',
              border: '1px solid var(--code-border-light)'
            }}>
              <code>conda install phytorch -c pytorch -c conda-forge</code>
            </pre>
          </div>

          <div style={{ marginBottom: '3rem' }}>
            <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem' }}>
              <strong>2. Fit a photosynthesis model:</strong>
            </h3>
            <pre className="code-block-bg" style={{
              padding: '1.5rem',
              borderRadius: '8px',
              fontSize: '0.95rem',
              lineHeight: '1.5',
              border: '1px solid var(--code-border-light)',
              overflow: 'auto'
            }}>
              <code>{`import torch
from phytorch.models import FvCB
from phytorch.fitting import fit_model

# Load your photosynthesis data
data = torch.load('aci_curve.pt')

# Initialize FvCB model
model = FvCB()

# Fit the model
result = fit_model(model, data)

print(f"Fitted Vcmax: {result.params['Vcmax']:.2f}")
print(f"Fitted Jmax: {result.params['Jmax']:.2f}")`}</code>
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <ReferencesSection />
        <GetStartedSection />
      </main>
    </Layout>
  );
}

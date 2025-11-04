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
    navigator.clipboard.writeText('pip install phytorch-lib');
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
            <code>pip install phytorch-lib</code>
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
            PhoTorch: a robust and generalized biochemical photosynthesis model fitting package based on PyTorch
          </Link>
          <pre className="code-block-bg" style={{
            padding: '1.5rem',
            borderRadius: '8px',
            fontSize: '0.9rem',
            lineHeight: '1.6',
            overflow: 'auto',
            border: '1px solid var(--code-border-light)'
          }}>
            <code>{`@article{lei2025photorch,
  title={PhoTorch: a robust and generalized biochemical photosynthesis model fitting package based on PyTorch},
  author={Lei, Tong and Rizzo, Kyle T and Bailey, Brian N},
  journal={Photosynthesis Research},
  volume={163},
  number={2},
  pages={21},
  year={2025},
  publisher={Springer}
}`}</code>
          </pre>
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
              <code>pip install phytorch-lib</code>
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
              <code>conda install phytorch-lib -c pytorch -c conda-forge</code>
            </pre>
          </div>

          <div style={{ marginBottom: '3rem' }}>
            <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem' }}>
              <strong>2. Fit any model with the unified API:</strong>
            </h3>
            <pre className="code-block-bg" style={{
              padding: '1.5rem',
              borderRadius: '8px',
              fontSize: '0.95rem',
              lineHeight: '1.5',
              border: '1px solid var(--code-border-light)',
              overflow: 'auto'
            }}>
              <code>{`from phytorch import fit
from phytorch.models.photosynthesis import FvCB
import pandas as pd

# Load your A-Ci curve data
df = pd.read_csv('aci_data.csv')

# Prepare data dictionary
data = {
    'Ci': df['Ci'].values,
    'Q': df['PARi'].values,
    'Tleaf': df['Tleaf'].values,
    'A': df['Photo'].values
}

# Fit the model (that's it!)
result = fit(FvCB(), data)

# View results
print(f"Vcmax25: {result.parameters['Vcmax25']:.2f}")
print(f"Jmax25: {result.parameters['Jmax25']:.2f}")
print(f"R² = {result.r_squared:.4f}")

# Plot comprehensive results (1:1, response curves, 3D surfaces)
result.plot()`}</code>
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
      title="PhyTorch"
      description="A unified Python toolkit for fitting plant physiology models">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <ReferencesSection />
        <GetStartedSection />
      </main>
    </Layout>
  );
}

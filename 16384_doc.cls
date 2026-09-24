% @file 16384_doc.cls
% @brief Implements a generic document class that 16-384 documents can all use
% to present a standard interface.
%
% @author Oscar Bezi (odb@andrew)
% @author Garrison Johnston (johnsg7@rpi.edu)
% @author Puneet Singhal (psinghal@andrew.cmu.edu)
%		Changes: added "\renewcommand{\thepartno}{\arabic{partno}}" to match the style with gradescope
%
% @since 12 February 2016
%===============================================================================

\LoadClass[12pt,addpoints]{exam}

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{16384_doc}[2024/09/10 Provides standard formatting for 16-384 handout documents.]

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{amsmath,amsfonts,amssymb}
\RequirePackage[left=1in, right=1in, top=1in, bottom=1in]{geometry}
\RequirePackage{graphicx}
\RequirePackage{lastpage}
\RequirePackage{etoolbox}
\RequirePackage{titlesec}
\RequirePackage{indentfirst}
\RequirePackage{enumitem}
\RequirePackage{titlesec}
\RequirePackage{nicefrac}
\RequirePackage{hyperref}

%% Hyperref setup
\usepackage{xcolor}
\usepackage{tcolorbox}
\hypersetup{
    colorlinks,
    linkcolor={red!50!black},
    citecolor={blue!50!black},
    urlcolor={blue!80!black}
}

% UPDATE THIS EACH SEMESTER
\newcommand{\semester}{Fall 2026}
\newcommand{\instructor}{\href{https://www.ri.cmu.edu/ri-faculty/jeffrey-ichnowski/}{Prof. Jeff Ichnowski}}
\newcommand{\taalbert}{\href{https://scholar.google.com/citations?user=9xBtRaMAAAAJ\&hl=en}{Albert Xu}}
\newcommand{\tahenry}{\href{https://s-hliao.github.io/}{Henry Liao}}
\newcommand{\takarthik}{\href{https://www.linkedin.com/in/srini-karthik/}{Karthik Srinivasan}}

% change the numbering of questions and subquestions
\renewcommand{\thepartno}{\arabic{partno}}
% Use a sans-serif font.
\renewcommand{\familydefault}{\sfdefault}
% A nice size for making grading sheet checkboxes out of a tabular.  \setlength{\tabcolsep}{20pt}
%\renewcommand{\arraystretch}{2}
\setlist[itemize]{noitemsep,nolistsep}

\newcommand{\sectionbreak}{\clearpage}

% The page header.
\lhead{Robot Kinematics and Dynamics}
\rhead{\assignmentname}
\headrule

% Uniform title format
\title{
    \assignmentname
    \ifdef{\writeup}{\\\textbf{Answer Key: Course Staff Only}}{}
}
\author{16-384: Robot Kinematics and Dynamics \\ \instructor \\ \taalbert \\ \tahenry \\ \takarthik}
\date{\ifdef{\duedate}{Due: 11:00AM EST, \duedate}{\examdate}}
\date{\ifdef{\duedatetime}{Due: \duedatetime}{}}

% Formatting for the assignment questions.
\bracketedpoints
\qformat{\thequestion) \thequestiontitle\hfill\textit{\thepoints}}

\ifdef{\writeup}{\printanswers}

% Definitions
\newcommand{\define}[2]{
\hangindent=2em
\hangafter=1
\noindent
\textbf{#1:}#2\\
}

% Written section for assignments
\newcommand{\writtenSection}{
\section{Instructions} 
\begin{itemize}
    \item The deadline for this assignment is 19th September, 2024 09:00 P.M.
    \item Zip your code into a single file named \texttt{<AndrewId>.zip}. See the complete submission checklist at the end, to ensure you have everything. Submit your PDF file to Gradescope.
    \item Each question (for points) is marked with a \textbf{points} heading.
    \item \textbf{Start early!} This homework may take a long time to complete.
    \item \textbf{During submission indicate the answer/page correspondence carefully when submitting on Gradescope.} If you skip a written question, just submit a blank page for it. This makes our work much easier to grade.
    \item If you have any questions or need clarifications, please post in Piazza or visit the TAs during the office hours.
    \item Unless otherwise specified, \textbf{all units are in radians, meters, and seconds}, where appropriate.
\end{itemize}


\end{questions}}



% Checklist
\newlist{checklist}{itemize}{2}
\setlist[checklist]{label=$\square$}

\newenvironment{submissionChecklist}
{
    \newpage
    \section{Submission Checklist}
    \begin{checklist}
}
{
    \end{checklist}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Macros for math shortcuts!

% Inline matrices.
\newcommand{\smcolvec}[1]{\ensuremath{\left[\begin{smallmatrix}#1\end{smallmatrix}\right]}}

% Larger matrices.
\newcommand{\colvec}[1]{\ensuremath{\begin{bmatrix}#1\end{bmatrix}}}

% Radians show up all the time.
\newcommand{\piover}[1]{\ensuremath{\frac{\pi}{#1}}}

% R^2 and such.
\newcommand\RR[1][]{\ifthenelse{\equal{#1}{}}{\ensuremath{\mathbb{R}}}{\ensuremath{\mathbb{R}^{#1}}}
}

% Shorthand for common variables.
\newcommand{\var}[2]{\ensuremath{#1_{#2}}}
\newcommand{\vard}[2]{\ensuremath{\dot{#1}_{#2}}}

\renewcommand\th[1][]{\var{\theta}{#1}}
\newcommand\thd[1][]{\vard{\theta}{#1}}

\renewcommand\l[1][]{\var{l}{#1}}
\newcommand\ld[1][]{\vard{l}{#1}}

\newcommand\x[1][]{\var{x}{#1}}
\newcommand\xd[1][]{\vard{x}{#1}}

\newcommand\y[1][]{\var{y}{#1}}
\newcommand\yd[1][]{\vard{y}{#1}}

\renewcommand\d[1][]{\var{d}{#1}}
\newcommand\dd[1][]{\vard{d}{#1}}

\newcommand\Q{\ensuremath{\mathbf{Q}}}
\newcommand\Qd{\ensuremath{\dot{\mathbf{Q}}}}

\newcommand\W{\ensuremath{\mathbf{W}}}
\newcommand\Wd{\ensuremath{\dot{\mathbf{W}}}}

% Standardize vectors and matrices.
\renewcommand{\vec}[1]{\ensuremath{\mathbf{#1}}}
\newcommand{\mat}[1]{\ensuremath{\mathbf{#1}}}
\renewcommand{\frame}[1]{\ensuremath{\mathbf{\{#1\}}}}

% Shorthand for inverses.
\newcommand{\inv}[1]{\ensuremath{\left(#1\right)^{-1}}}

% Nice transposes.
\newcommand{\trans}[1]{\ensuremath{\left(#1\right)^{\top}}}

% Homogeneous transforms.
\newcommand{\transform}[3]{\mat{#1_{#2}^{#3}}}
\renewcommand{\H}[2]{\transform{H}{#1}{#2}}
\newcommand{\R}[2]{\transform{R}{#1}{#2}}
\renewcommand{\t}[2]{\transform{d}{#1}{#2}}

\newcommand{\p}[1]{\transform{p}{}{#1}}
\renewcommand{\v}[1]{\transform{v}{}{#1}}
\newcommand{\f}[1][]{\ensuremath{f_{#1}}}

% Jacobians
\newcommand{\J}{\mat{J}}
\newcommand{\Jt}{\trans{\J}}

% Partial derivatives.
\newcommand{\partialfrac}[2]{\ensuremath{\frac{\partial #1}{\partial #2}}}

%Vector Norms
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}

% Rank
\DeclareMathOperator{\rank}{rank}

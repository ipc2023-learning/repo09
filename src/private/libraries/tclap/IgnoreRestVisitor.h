// -*- Mode: c++; c-basic-offset: 4; tab-width: 4; -*-

/******************************************************************************
 *
 *  file:  IgnoreRestVisitor.h
 *
 *  Copyright (c) 2003, Michael E. Smoot .
 *  Copyright (c) 2020, Google LLC
 *  All rights reserved.
 *
 *  See the file COPYING in the top directory of this distribution for
 *  more information.
 *
 *  THE SOFTWARE IS PROVIDED _AS IS_, WITHOUT WARRANTY OF ANY KIND, EXPRESS
 *  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef TCLAP_IGNORE_REST_VISITOR_H
#define TCLAP_IGNORE_REST_VISITOR_H

#include "CmdLineInterface.h"
#include "Visitor.h"

namespace TCLAP
{
    /**
     * A Visitor that tells the CmdLine to begin ignoring arguments after
     * this one is parsed.
     */
    class IgnoreRestVisitor : public Visitor
    {
      public:
        IgnoreRestVisitor(CmdLineInterface& cmdLine) : Visitor(), cmdLine_(cmdLine) {}
        void visit() { cmdLine_.beginIgnoring(); }

      private:
        CmdLineInterface& cmdLine_;
    };
}  // namespace TCLAP

#endif  // TCLAP_IGNORE_REST_VISITOR_H
